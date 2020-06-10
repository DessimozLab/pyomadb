'''
    PyOMADB -- client to the OMA browser, using the REST API.

    (C) 2018 Alex Warwick Vesztrocy <alex@warwickvesztrocy.co.uk>

    This file is part of PyOMADB.

    PyOMADB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyOMADB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with PyOMADB.  If not, see <http://www.gnu.org/licenses/>.
'''
from collections import defaultdict
from functools import lru_cache
from io import StringIO
from pprint import pformat
from property_manager import lazy_property
from requests_cache.core import CachedSession
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import quote as uri_quote
from tqdm import tqdm
import appdirs
import json
import numbers
import os
import pandas as pd
import random
import requests
import shutil
import warnings


class AttrDict(object):
    def __init__(self, d):
        self.__dictionary__ = d
        self.get = self.__dictionary__.get
        self.items = self.__dictionary__.items
        self.keys = self.__dictionary__.keys
        self.pop = self.__dictionary__.pop
        self.values = self.__dictionary__.values

        self._setup()

    def _setup(self):
        for (k, v) in self.__dictionary__.items():
            if isinstance(v, dict):
                self.__dictionary__[k] = self._recurse_create(v)
            elif isinstance(v, list):
                self.__dictionary__[k] = [(x if not isinstance(x, dict) else
                                           self._recurse_create(x))
                                          for x in v]
            if hasattr(self, '_setup_extra'):
                self._setup_extra(k, v)

    def __len__(self):
        return len(self.__dictionary__)

    def __getitem__(self, i):
        return self.__dictionary__[i]

    def __setitem__(self, i, d):
        self.__dictionary__[i] = d

    def __dir__(self):
        return self.__dictionary__.keys()

    def __getattr__(self, attr):
        try:
            return self.__dictionary__[attr]
        except KeyError:
            return self.__getattribute__(attr)

    def _recurse_create(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def __repr__(self):
        return pformat(self.__dictionary__)

    def __str__(self):
        return repr(self.__dictionary__)

    def _undo_attrdict(self):
        # This is for strange scenarios where there isn't too much data,
        # as it creates a copy.
        y = {}
        for (k, v) in self.items():
            if isinstance(v, AttrDict):
                y[k] = v._undo_attrdict()
            else:
                y[k] = v
        return y


class ClientResponse(AttrDict):
    def __init__(self, response, client=None, _is_paginated=None):
        self.client = client
        super().__init__(response)

    def _recurse_create(self, *args, **kwargs):
        return type(self)(*args, client=self.client, **kwargs)

    def __getattr__(self, attr):
        r = super().__getattr__(attr)
        return r if not isinstance(r, ClientRequest) else r()

    def _setup_extra(self, k, v):
        if isinstance(v, str) and v.startswith(self.client.endpoint):
            self.__dictionary__[k] = ClientRequest(self.client, v)

    def as_dataframe(self, k):
        '''
        Get a dataframe for the list stored in a particular key / attribute.

        Note: any nested attribute-dictionaries shall have to be converted to
        dictionaries for compatibility reasons with pandas.

        :return: data frame for list of entries
        :rtype: pd.DataFrame
        '''
        if not isinstance(self[k], list):
            raise TypeError('{} is not a list, cannot convert to '
                            'dataframe.'.format(k))
        if not isinstance(self[k][0], ClientResponse):
            raise TypeError('{} is not a list of dictionary-like elements, '
                            'cannot convert to dataframe.'.format(k))
        return pd.DataFrame.from_records(map(lambda x: x._undo_attrdict(),
                                             self[k]))


class ClientPagedResponse(object):
    def __init__(self, client, response, progress_desc=None):
        self.client = client
        self.response = response
        self.progress_desc = ('' if progress_desc is None else progress_desc)

    def _yield_elts(self, f, x, pbar):
        for e in map(f, x):
            yield e
            pbar.update()

    def __iter__(self):
        '''
        Iterates over all entries, silently lazily loading the next page when
        required.
        '''
        r = self.response
        x = json.loads(r.content)

        pbar = tqdm(desc=self.progress_desc,
                    unit=' entries',
                    total=int(r.headers.get('X-Total-Count', len(x))),
                    disable=(len(self.progress_desc) == 0))

        yield from self._yield_elts(lambda e: ClientResponse(e,
                                                             client=self.client), 
                                    x, pbar)

        while 'next' in r.links:
            r = self.client._request(uri=r.links['next']['url'], raw=True)
            yield from self._yield_elts(lambda e: ClientResponse(e,
                                                                 client=self.client), 
                                        json.loads(r.content), pbar)
        pbar.close()

    def as_dataframe(self):
        '''
        Retrieves all entries, from all pages. Returns as pandas data frame.

        Note: in general, it would be better to use iter_dataframes to deal
        with the returned entries in chunks (if possible).

        :return: data frame containing all entries in response
        :rtype: pd.DataFrame
        '''
        return pd.concat(self.iter_dataframes())

    def iter_dataframes(self):
        '''
        Yields dataframes for each page of response. That is, entries are
        yielded in chunks.
        '''
        r = self.response
        x = json.loads(r.content)

        pbar = tqdm(desc=self.progress_desc,
                    unit=' entries',
                    total=int(r.headers.get('X-Total-Count', len(x))),
                    disable=(len(self.progress_desc) == 0))

        yield pd.DataFrame.from_records(
            self._yield_elts(self.client._add_lazy_calls,
                             x,
                             pbar))

        while 'next' in r.links:
            r = self.client._request(uri=r.links['next']['url'], raw=True)
            yield pd.DataFrame.from_records(
                self._yield_elts(self.client._add_lazy_calls,
                                 json.loads(r.content),
                                 pbar))
        pbar.close()


class ClientRequest(object):
    def __init__(self, client, uri):
        self.client = client
        self.uri = uri

    def __call__(self):
        return self.client._request(uri=self.uri)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '<API Request {}>'.format(self.uri[len(self.client.endpoint):])


class ClientException(Exception):
    pass


class ClientTimeout(Exception):
    pass


class NotFound(Exception):
    pass


class Client(object):
    '''
    Client for the OMA browser REST API.

    Initialisation example::

        from omadb import Client
        c = Client()

    :raises ClientException: for 400, 404, 500 errors.
    :raises ClientTimeout: for timeout when interacting with REST endpoint.
    '''
    HEADERS = {'Content-type': 'application/json',
               'Accept': 'application/json'}
    TIMEOUT = 60
    PER_PAGE = 10000
    RAMCACHE_SIZE = 10000

    def __init__(self, endpoint='omabrowser.org/api', persistent_cached=False,
                 persistent_cache_path=None):
        '''
        :param str endpoint: OMA REST API endpoint (default omabrowser.org/api)
        :param bool persistent_cached: whether to cache queries on disk in SQLite DB.
        :param persistent_cache_path: location for persistent cache, optional
        :type persistent_cache_path: str or None
        '''
        self.endpoint = ('https://' + endpoint
                         if not endpoint.startswith('http')
                         else endpoint)
        from . import __version__ as client_version
        self.HEADERS['User-agent'] = 'pyomadb/'+client_version

        self.persistent_cached = persistent_cached
        if self.persistent_cached:
            if persistent_cache_path is None:
                self.CACHE_PATH = appdirs.user_cache_dir('py' + __package__)
            else:
                self.CACHE_PATH = os.path.abspath(persistent_cache_path)
            self._version_check()
            self._setup_cache()
        self._setup()

        self._temp_path = TemporaryDirectory()

    def clear_cache(self):
        '''
        Clear both RAM and persistent cache.
        '''
        self._request_get.cache_clear()
        if hasattr(self, 'session'):
            self.session.close()
            del self.session
            shutil.rmtree(self.CACHE_PATH)
            if self.persistent_cached:
                self._version_check()
                self._setup_cache()

    def _setup_cache(self):
        os.makedirs(self.CACHE_PATH, exist_ok=True)
        self.session = CachedSession(cache_name=os.path.join(self.CACHE_PATH,
                                                             'api-cache'),
                                     backend='sqlite')

    def _setup(self):
        self.genomes = Genomes(self)
        self.entries = self.proteins = Entries(self)
        self.hogs = HOGs(self)
        self.groups = OMAGroups(self)
        self.function = Function(self)
        self.taxonomy = Taxonomy(self)
        self.pairwise = PairwiseRelations(self)
        self.xrefs = self.external_references = ExternalReferences(self)

    @lazy_property
    def version(self):
        return self._request(action='version')

    @lazy_property
    def oma_release(self):
        return self.version['oma_version']

    @lazy_property
    def api_version(self):
        return self.version['api_version']

    def _version_check(self):
        db_fn = os.path.join(self.CACHE_PATH, 'oma-version')
        api_fn = os.path.join(self.CACHE_PATH, 'api-version')

        if os.path.isdir(self.CACHE_PATH):
            with open(db_fn, 'rt') as fp:
                oma_release = next(fp).rstrip()
            with open(api_fn, 'rt') as fp:
                api_version = next(fp).rstrip()

            if ((self.oma_release != oma_release) or
                    (self.api_version != api_version)):
                warnings.warn('OMA database and / or API updated. '
                              'Clearing cache.')
                shutil.rmtree(self.CACHE_PATH)

        os.makedirs(self.CACHE_PATH, exist_ok=True)
        if not os.path.isfile(db_fn):
            with open(db_fn, 'wt') as fp:
                fp.write(self.oma_release + '\n')
        if not os.path.isfile(api_fn):
            with open(api_fn, 'wt') as fp:
                fp.write(self.api_version + '\n')

    def _get_request_uri(self, uri=None, action=None, subject=None,
                         params=None, **kwargs):
        if uri is None:
            if action is None or (isinstance(action, list) and len(action) == 0):
                raise Exception('No action declared.')

            if isinstance(action, list):
                uri = '/{}'.format(action[0])
                uri += '/{}'.format(subject) if subject is not None else ''
                if len(action) > 1 and action[1] != 'list':
                    uri += '/{}'.format(action[1])
            else:
                uri = '/{}'.format(action)
                uri += '/{}'.format(subject) if subject is not None else ''
            uri = uri_quote(uri)

        uri = uri + '/' if (not uri.endswith('/') and '?' not in uri) else uri

        # Parse params
        params = {} if params is None else params
        paginated = kwargs.pop('paginated', False)
        if paginated:
            params['per_page'] = self.PER_PAGE

        # Add the rest of kwargs to params
        params.update(kwargs)

        return (str(uri), params)

    def _get_request_data(self, data=None, **kwargs):
        if not isinstance(data, dict):
            raise ValueError('Data is not defined.')
        return json.dumps(data)

    @lru_cache(RAMCACHE_SIZE)
    def _request_get(self, url, **params):
        get = getattr(self, 'session', requests).get
        return get(url,
                   headers=self.HEADERS,
                   params=params,
                   timeout=self.TIMEOUT)

    def _request_post(self, url, data):
        return requests.post(url,
                             data=data,
                             headers=self.HEADERS,
                             timeout=self.TIMEOUT)

    def _request(self, request_type='get', **kwargs):
        raw = kwargs.pop('raw', False)
        progress_desc = kwargs.pop('progress_desc', '')
        as_dataframe = kwargs.pop('as_dataframe', False)

        # Get URI and params
        (uri, params) = self._get_request_uri(**kwargs)
        url = ((self.endpoint + uri) if not uri.startswith(self.endpoint)
               else uri)
        try:
            if request_type is 'get':
                r = self._request_get(url, **params)
            elif request_type is 'post':
                data = self._get_request_data(**kwargs)
                r = self._request_post(url, data)
            else:
                raise ValueError('Unsure how to deal with request type'
                                 '{}'.format(request_type))
        except requests.exceptions.Timeout:
            raise ClientTimeout('API timed out after'
                                '{}s.'.format(self.TIMEOUT))

        # Check if response OK
        response_status = True
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            response_status = str(e).split('for url:')[0]

        if response_status is True:
            if raw:
                return r

            elif kwargs.get('paginated', False) or self._is_paginated(r):
                return ClientPagedResponse(self,
                                           r,
                                           progress_desc=progress_desc)

            else:
                content = json.loads(r.content)
                if isinstance(content, list):
                    if as_dataframe:
                        return pd.DataFrame.from_records(
                            map(self._add_lazy_calls, content))
                    else:
                        return list(map(lambda x: ClientResponse(x, client=self),
                                        content))
                else:
                    return ClientResponse(content, client=self)
        else:
            if r.status_code in {400, 404}:
                try:
                    content = json.loads(r.content)
                except json.JSONDecodeError:
                    content = {}

                z = ClientResponse(content, client=self)
                if set(z.keys()) == {'detail'}:
                    response_status += '["' + z.detail + '"]'
            raise ClientException(response_status)

    def _is_paginated(self, r):
        return len(set(r.links.keys()) & {'next', 'last'}) == 2

    def _add_lazy_calls(self, e):
        for (k, v) in e.items():
            if isinstance(v, str) and v.startswith(self.endpoint):
                e[k] = ClientRequest(self, v)
            elif isinstance(v, dict):
                e[k] = self._add_lazy_calls(v)
        return e


class CoronaClient(Client):
    '''
    Client for the Corona OMA browser REST API.

    Initialisation example::

        from omadb import CoronaClient
        c = CoronaClient()

    :raises ClientException: for 400, 404, 500 errors.
    :raises ClientTimeout: for timeout when interacting with REST endpoint.
    '''
    def __init__(self, endpoint='corona.omabrowser.org/api', persistent_cached=False,
                 persistent_cache_path=None):
        '''
        :param str endpoint: OMA REST API endpoint (default corona.omabrowser.org/api)
        :param bool persistent_cached: whether to cache queries on disk in SQLite DB.
        :param persistent_cache_path: location for persistent cache, optional
        :type persistent_cache_path: str or None
        '''
        super().__init__(endpoint=endpoint,
                         persistent_cached=persistent_cached,
                         persistent_cache_path=persistent_cache_path)


class ClientFunctionSet(object):
    def __init__(self, client):
        self._client = client
        if hasattr(self, '_setup'):
            self._setup()


class Genomes(ClientFunctionSet):
    '''
    API functionality for genome information.

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        wheat = c.genomes.genome('WHEAT')
    '''
    def __getitem__(self, genome_id):
        '''
        Retrieve information on a genome in OMA.

        :param genome_id: unique identifier for genome, NCBI taxonomic ID or UniProt species code
        :type genome_id: str or int

        :return: genome information
        :rtype: ClientResponse
        '''
        return self.genome(genome_id)

    def __iter__(self):
        '''Iterate over all genomes.''' 
        yield from self.genomes.items()

    @property
    def list(self):
        '''
        Synonym for `genomes`. Retrieve information on all genomes in OMA.

        :return: information on all genomes
        :rtype: ClientResponse
        '''
        return self.genomes

    @lazy_property
    def genomes(self):
        '''
        Retrieve information on all genomes in OMA.

        :return: information on all genomes
        :rtype: ClientResponse
        '''
        r = {sp.pop('code'): sp
             for sp in self._client._request(action='genome',
                                             paginated=True)}
        return ClientResponse(r, client=self._client)

    def as_dataframe(self):
        '''
        Retrieve information on all genomes in OMA, return as pandas data
        frame.

        :return: information on all genomes
        :rtype: pd.DataFrame
        '''
        return self._client._request(action='genome',
                                     paginated=True).as_dataframe()

    def genome(self, genome_id):
        '''
        Retrieve information on a genome in OMA.

        :param genome_id: unique identifier for genome, NCBI taxonomic ID or UniProt species code
        :type genome_id: str or int

        :return: genome information
        :rtype: ClientResponse
        '''
        return self._client._request(action='genome', subject=genome_id)

    def proteins(self, genome_id, progress=False):
        '''
        Retrive all proteins for a particular genome.

        :param genome_id: unique identifier for genome, NCBI taxonomic ID or UniProt species code
        :type genome_id: str or int
        :param bool progress: whether to show progress bar

        :return: proteins in genome
        :rtype: ClientPagedResponse
        '''
        desc = 'Retrieving proteins for {}'.format(sp_code) if progress else ''
        return self._client._request(action=['genome', 'proteins'],
                                     subject=genome_id,
                                     paginated=True,
                                     progress_desc=desc)


class HOGs(ClientFunctionSet):
    '''
    API functionality for HOG information.

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        entry = c.hogs['WHEAT00001']
    '''
    def _ensure_hog_id(self, hog_id):
        return 'HOG:{:07d}'.format(hog_id) if isinstance(hog_id, numbers.Number) else hog_id

    def __getitem__(self, hog_id):
        '''
        Retrieve the detail available for a given HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int

        :return: HOG information
        :rtype: ClientResponse
        '''
        return self.info(hog_id)

    def list(self):
        '''
        Retrieve information about all HOGs.

        :return: all HOGs
        :rtype: ClientPagedResponse
        '''
        return self.at_level(level=None)

    def at_level(self, level):
        '''
        Retrieve list of HOGs at a particular level

        :param str level: level of interest
       
        :return: all hogs at a particular level
        :rtype: ClientPagedResponse
        '''
        return self._client._request(action='hog', 
                                     params={'level': level},
                                     paginated=True)

    def info(self, hog_id):
        '''
        Retrieve the detail available for a given HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int

        :return: HOG information
        :rtype: ClientResponse
        '''
        return self._client._request(action='hog',
                                     subject=self._ensure_hog_id(hog_id))[0]

    def members(self, hog_id, level=None, as_dataframe=None):
        '''
        Retrieve list of protein entries in a given HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int
        :param str level: level of interest
        :param bool as_dataframe: whether to return as pandas data frame, optional

        :return: list of members
        :rtype: list or pd.DataFrame
        '''
        z = self._client._request(action=['hog', 'members'],
                                  subject=self._ensure_hog_id(hog_id),
                                  level=level)
        if as_dataframe:
            return z.as_dataframe('members')
        else:
            return z.members

    def external_references(self, hog_id, type=None):
        '''
        Retrieve external references for all members of a particular HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int
        :param str level: level of interest

        :return: external references
        :rtype: dict
        '''
        return self.xrefs(hog_id, type=type)

    def xrefs(self, hog_id, level=None, type=None, as_dataframe=None):
        '''
        Retrieve external references for all members of a particular HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int
        :param str level: level of interest
        :param bool as_dataframe: whether to return as pandas data frame, optional

        :return: external references
        :rtype: dict or pd.DataFrame
        '''
        if as_dataframe:
            if type is not None:
                return pd.DataFrame.from_records({'omaid': m['omaid'],
                                                  'xref': x}
                                                 for m in self.members(hog_id, level=level)
                                                 for x in self._client.entries.xrefs(m['entry_nr'], type))
            else:
                return pd.DataFrame.from_records({'omaid': m['omaid'],
                                                  'type': xtype,
                                                  'xref': x}
                                                 for m in self.members(hog_id, level=level)
                                                 for (xtype, xs) in
                                                 self._client.entries.xrefs(m['entry_nr']).items()
                                                 for x in xs)
        else:
            return {m['omaid']: self._client.entries.xrefs(m['entry_nr'], type)
                    for m in self.members(hog_id, level=level)}

    def analyze(self, hog_id):
        '''
        Use the PyHAM package to analyse a particular hierarchical orthologous
        group.
        
        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int

        :return: analysis object
        :rtype: pyham.Ham
        '''
        return self.analyse(hog_id)

    def analyse(self, hog_id):
        '''
        Use the PyHAM package to analyse a particular hierarchical orthologous
        group.
        
        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int

        :return: analysis object
        :rtype: pyham.Ham
        '''
        try:
            from pyham import Ham
        except ImportError:
            raise ImportError('For HOG analysis, the pyham package is '
                              'required.')

        z = self[self[hog_id].roothog_id]
        m = z.members_url.members[0].omaid
        root_level = z.level

        try:
            r = requests.get('https://omabrowser.org/oma/hogs/{}/orthoxml'.format(m),
                             timeout=self._client.TIMEOUT)
        except requests.exceptions.Timeout:
            raise ClientTimeout('OrthoXML request timed out after'
                                '{}s.'.format(self.TIMEOUT))

        xml = self._client.taxonomy.get(members=[root_level],
                                        format='phyloxml').decode('ascii')
        t = StringIO(xml)
        return Ham(tree_file=t,
                   hog_file=r.content.decode('utf-8'),
                   orthoXML_as_string=True,
                   tree_format='phyloxml',
                   use_internal_name=True)

    def iham(self, hog_id):
        '''
        Create an iHam page and print path to temporary file.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int
        '''
        z = self.analyse(hog_id)
        iham = z.create_iHam(list(z.top_level_hogs.values())[0])

        while True:
            fn = os.path.join(self._client._temp_path.name,
                              'iham{:06d}.html'.format(random.randint(0,999999)))
            if not os.path.isfile(fn):
                break
        with open(fn, 'wt') as fp:
            fp.write(iham.HTML)
        print('{}'.format(fn))


class Entries(ClientFunctionSet):
    '''
    API functionality for protein entries.

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        entry = c.entries['WHEAT00001']
    '''
    def __getitem__(self, entry_id):
        '''
        Retrieve the information available for a protein entry.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int

        :return: entry information
        :rtype: ClientResponse
        '''
        return self.info(entry_id)

    def info(self, entry_id):
        '''
        Retrieve the information available for a protein entry.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int

        :return: entry information
        :rtype: ClientResponse
        '''
        if isinstance(entry_id, numbers.Number) or isinstance(entry_id, str):
            return self._client._request(action='protein', subject=entry_id)
        else:
            return self._client._request(action='protein',
                                         subject='bulk_retrieve',
                                         data={'ids': list(entry_id)},
                                         request_type='post')

    def domains(self, entry_id):
        '''
        Retrieve the domains present in a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int

        :return: domain information
        :rtype: ClientResponse
        '''
        return self._client._request(action=['protein', 'domains'],
                                     subject=entry_id)

    @lazy_property
    def _gene_ontology(self):
        from goatools.obo_parser import GODag
        url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
        try:
            r = requests.get(url, timeout=self._client.TIMEOUT)
        except requests.exceptions.Timeout:
            raise ClientTimeout('Gene Ontology download request '
                                'timed out after '
                                '{}s.'.format(self.TIMEOUT))
        with NamedTemporaryFile(suffix='.obo') as fp:
            fp.write(r.content)
            return GODag(fp.name)

    def gene_ontology(self, entry_id, aspect=None, as_dataframe=None,
                      as_goatools=None, progress=False, **kwargs):
        '''
        Retrieve any associations to Gene Ontology terms for a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int or list
        :param str aspect: GO aspect - biological process (BP), cellular component (CC), molecular function (MF)
        :param bool as_dataframe: whether to return as pandas data frame, optional
        :param bool as_goatools: whether to return as GOATOOLS GOEA object, optional
        :param bool progress: whether to show a progress bar during load (default False)

        :return: gene ontology associations
        :rtype: list or pd.DataFrame or goatools.go_enrichment.GOEnrichmentStudy
        '''
        if isinstance(entry_id, list):
            assert (not (as_goatools and as_dataframe)), 'Cannot load both GOATOOLS and data frame!'

            if as_dataframe:
                dfs = []
                for x in tqdm(entry_id, desc='Retrieving GO',
                              disable=(not progress)):
                    df = self.gene_ontology(x, aspect=aspect, as_dataframe=True)
                    df['query_id'] = x
                    df = df.set_index('query_id')
                    dfs.append(df)

                return pd.concat(dfs)
            else:
                z = {x: self.gene_ontology(x, aspect=aspect)
                     for x in tqdm(entry_id,
                                   desc='Retrieving GO',
                                   disable=(not progress))}

                if as_goatools:
                    from goatools.go_enrichment import GOEnrichmentStudy
                    
                    goea = GOEnrichmentStudy(z.keys(),
                                             {k: {x.GO_term for x in v}
                                              for (k, v) in z.items()},
                                             self._gene_ontology,
                                             methods=['fdr_bh'])
                    return goea
                else:
                    return z

        else:
            if as_goatools:
                raise ValueError('Not possible to load GOEA object for single '
                                 'entry.')
            z = self._client._request(action=['protein', 'gene_ontology'],
                                      subject=entry_id,
                                      as_dataframe=as_dataframe)
            if aspect is None:
                return z
            
            # Translate
            aspects = {'bp': 'biological_process',
                       'cc': 'cellular_component',
                       'mf': 'molecular_function'}
            valid_aspects = set(aspects.values())
            aspect = aspects.get(aspect.lower(), aspect)

            assert (aspect in valid_aspects), 'Unknown aspect: {}'.format(aspect)
            if as_dataframe:
                return z[z.aspect == aspect]
            else:
                return list(filter(lambda x: x.aspect == aspect, z))

    def homoeologs(self, entry_id):
        '''
        Retrieve all homoeologs for a given protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int

        :return: list of homoeologs
        :rtype: ClientResponse
        '''
        return self.homoeologues(entry_id)

    def homoeologues(self, entry_id):
        '''
        Retrieve all homoeologues for a given protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int

        :return: list of homoeologues
        :rtype: ClientResponse
        '''
        return self._client._request(action=['protein', 'homoeologs'],
                                     subject=entry_id)

    def orthologs(self, entry_id, rel_type=None):
        '''
        Retrieve list of all identified orthologs of a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int
        :param rel_type: relationship type to filter to ('1:1', '1:many', 'many:1', or 'many:many')
        :type rel_type: str or None

        :return: list of orthologs
        :rtype: ClientResponse
        '''
        return self.orthologues(entry_id, rel_type=rel_type)

    def orthologues(self, entry_id, rel_type=None):
        '''
        Retrieve list of all identified orthologues of a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int
        :param rel_type: relationship type ('1:1', '1:many', 'many:1', or 'many:many'), optional
        :type rel_type: str or None

        :return: list of orthologues
        :rtype: ClientResponse
        '''
        return self._client._request(action=['protein', 'orthologs'],
                                     subject=entry_id,
                                     params=({'rel_type': rel_type}
                                             if rel_type is not None else None))

    def cross_references(self, entry_id, type=None):
        '''
        Retrieve all cross-references for a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int
        :param type: specify type of cross-references to retain
        :type type: str or None

        :return: cross references
        :rtype: dict or set
        '''
        return self.xrefs(entry_id, type=type)

    def xrefs(self, entry_id, type=None):
        '''
        Retrieve all cross-references for a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int
        :param type: specify type of cross-references to retain
        :type type: str or None

        :return: cross references
        :rtype: dict or set
        '''
        def reformat(r):
            z = defaultdict(list)
            for x in r:
                z[x['source']].append(x['xref'])
            return z

        r = self._client._request(action=['protein', 'xref'],
                                  subject=entry_id)
        if type is not None:
            return set(map(lambda x: x['xref'],
                           filter(lambda x: x['source'] == type, r)))
        else:
            return reformat(r)

    def search(self, sequence, search=None, full_length=None):
        '''
        Search for closest sequence in OMA database.

        :param str query: query sequence
        :param search: search strategy ('exact, 'approximate', 'mixed' [Default])
        :type search: str or None
        :param bool full_length: indicates if exact matches have to be full length (by default, not)

        :return: closest entries
        :rtype: ClientResponse
        '''
        params = {'query': sequence,
                  'full_length': bool(full_length)}

        search_valid = {'exact', 'approximate', 'mixed'}
        search = search.lower() if search is not None else None
        if (search not in search_valid and search is not None):
            raise ValueError('{} is not a valid search method. Choose from {}'
                             .format(search, ', '.join(search_valid)))
        elif search is not None:
            params['search'] = search

        return self._client._request(action='sequence', params=params)


class Function(ClientFunctionSet):
    '''
    API functionality for retrieving functional annotations for sequences.

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        gos = c.function('ATCATATCAT')
    '''
    def __call__(self, seq, as_dataframe=None):
        '''
        Annotate a sequence with GO terms based on annotations stored in the
        OMA database.

        :param str query: query sequence
        :param bool as_dataframe: whether to return as pandas data frame, optional

        :return: results of fast function prediction
        :rtype: list or pd.DataFrame
        '''
        return self._client._request(action='function',
                                     params={'query': seq},
                                     as_dataframe=as_dataframe)


class OMAGroups(ClientFunctionSet):
    '''
    API functionality for retrieving information on OMA groups.

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        og = c.groups['WHEAT00001']
    '''
    def __getitem__(self, group_id):
        '''
        Retrieve information available for a given OMA group.

        :param group_id: unique identifier of a group - either group number, fingerprint or entry ID of a member.
        :type group_id: int or str

        :return: group information
        :rtype: ClientResponse
        '''
        return self.info(group_id)

    def __iter__(self):
        '''
        Iterate over all OMA groups in the current release.

        :yields: groups
        '''
        yield from self._client._request(action='group',
                                         paginated=True)

    def info(self, group_id):
        '''
        Retrieve information available for a given OMA group.

        :param group_id: unique identifier of a group - either group number, fingerprint or entry ID of a member.
        :type group_id: int or str

        :return: group information
        :rtype: ClientResponse
        '''
        return self._client._request(action='group', subject=group_id)

    def close_groups(self, group_id, as_dataframe=None):
        '''
        Retrieve the sorted list of closely related groups for a given OMA
        group.

        :param group_id: unique identifier of a group - either group number, fingerprint or entry ID of a member.
        :type group_id: int or str
        :param bool as_dataframe: whether to return as pandas data frame, optional

        :return: sorted list of closely related groups
        :rtype: list or pd.DataFrame
        '''
        return self._client._request(action=['group', 'close_groups'],
                                     subject=group_id,
                                     as_dataframe=as_dataframe)


class Taxonomy(ClientFunctionSet):
    def get(self, members=None, format=None, collapse=None):
        '''
        Retrieve taxonomy in a particular format and return as string.

        :param list members: list of members to get the induced taxonomy for, optional
        :param str format: format of the taxonomy (dictionary [default], newick or phyloxml)
        :param bool collapse: whether or not to collapse levels with single child, optional (default yes)

        :return: taxonomy
        :rtype: str
        '''
        members = (','.join(members) if members is not None else None)
        x = self._client._request(action='taxonomy',
                                  params={'members': members,
                                          'type': format,
                                          'collapse': collapse},
                                 raw=((format is not None) and
                                      (format is not 'dictionary')))

        if format is None or format is 'dictionary':
            return x
        elif format is 'newick':
            return json.loads(x.content)['newick']
        elif format is 'phyloxml':
            return x.content
        else:
            # This shouldn't really happen (Dec 2018)
            return x

    def read(self, root, format=None, collapse=None):
        '''
        Retrieve taxonomy in a particular format and return as string.

        :param root: taxon ID, species name or UniProt species code for root taxonomic level, optional
        :type root: str or int
        :param str format: format of the taxonomy (dictionary [default], newick or phyloxml)
        :param bool collapse: whether or not to collapse levels with single child, optional (default yes)

        :return: taxonomy 
        :rtype: str

        '''
        x = self._client._request(action=['taxonomy', root],
                                  params={'type': format,
                                          'collapse': collapse},
                                  raw=((format is not None) and 
                                       (format is not 'dictionary')))

        if format is None or format is 'dictionary':
            return x
        elif format is 'newick':
            return json.loads(x.content)['newick']
        elif format is 'phyloxml':
            return x.content
        else:
            # This shouldn't really happen (Dec 2018)
            return x

    def dendropy_tree(self, members=None, root=None, with_names=None):
        '''
        Retrieve taxonomy and load as dendropy tree.

        :param list members: list of members to get the induced taxonomy for, optional
        :param root: taxon ID, species name or UniProt species code for root taxonomic level, optional
        :type root: str or int or None
        :param bool with_names: whether to use species code (False, default) or species names (True), optional

        :return: taxonomy loaded as dendropy tree object
        :rtype: dendropy.Tree
        '''
        try:
            import dendropy
        except ImportError:
            raise ImportError('Optional dependency of dendropy not installed.')

        def get_names(tree, with_names):
            internal_names = [tree.name]
            leaf_names = []
            nodes = tree.children
            while len(nodes) > 0:
                nodes1 = []
                for n in nodes:
                    if hasattr(n, 'children'):
                        nodes1 += n.children
                        internal_names.append(n.name)
                    elif with_names:
                        leaf_names.append(n.name)
                    else:
                        leaf_names.append(n.code)
                nodes = nodes1
            return (internal_names + leaf_names)

        if ((members is not None) and (root is not None)):
            raise ValueError('Taxonomy undefined in API when members and '
                             'root are both set.')
        elif members is not None:
            struct = self.get(members=members)
        elif root is not None:
            struct = self.read(root)
        else:
            struct = self.get()

        taxon_namespace = dendropy.TaxonNamespace(get_names(struct,
                                                            with_names))
        tree = dendropy.Tree(taxon_namespace=taxon_namespace)

        tree.seed_node.taxon = taxon_namespace.get_taxon(struct.name)
        nodes = [(tree.seed_node, c) for c in struct.children]
        while len(nodes) > 0:
            nodes1 = []
            for (parent, child) in nodes:
                ch = parent.new_child(edge_length=1)
                if hasattr(child, 'children'):
                    ch.taxon = taxon_namespace.get_taxon(child.name)
                    nodes1 += [(ch, c) for c in child.children]
                elif with_names:
                    ch.taxon = taxon_namespace.get_taxon(child.name)
                else:
                    ch.taxon = taxon_namespace.get_taxon(child.code)
            nodes = nodes1

        return tree

    def ete_tree(self, members=None, root=None, with_names=None):
        '''
        Retrieve taxonomy and load as ete3 tree.

        :param list members: list of members to get the induced taxonomy for, optional
        :param root: taxon ID, species name or UniProt species code for root taxonomic level, optional
        :type root: str or int or None
        :param bool with_names: whether to use species code (False, default) or species names (True), optional

        :return: taxonomy loaded as ete tree object
        :rtype: ete.Tree
        '''
        try:
            import ete3
        except ImportError:
            raise ImportError('Optional dependency of ete3 not installed.')

        def get_names(tree, with_names):
            internal_names = [tree.name]
            leaf_names = []
            nodes = tree.children
            while len(nodes) > 0:
                nodes1 = []
                for n in nodes:
                    if hasattr(n, 'children'):
                        nodes1 += n.children
                        internal_names.append(n.name)
                    elif with_names:
                        leaf_names.append(n.name)
                    else:
                        leaf_names.append(n.code)
                nodes = nodes1
            return (internal_names + leaf_names)

        if ((members is not None) and (root is not None)):
            raise ValueError('Taxonomy undefined in API when members and '
                             'root are both set.')
        elif members is not None:
            struct = self.get(members=members)
        elif root is not None:
            struct = self.read(root)
        else:
            struct = self.get()

        tree = ete3.Tree()
        tree.name = struct.name
        nodes = [(tree, c) for c in struct.children]
        while len(nodes) > 0:
            nodes1 = []
            for (parent, child) in nodes:
                ch = parent.add_child(dist=1)
                if hasattr(child, 'children'):
                    ch.name = child.name
                    nodes1 += [(ch, c) for c in child.children]
                elif with_names:
                    ch.name = child.name
                else:
                    ch.name = child.code
            nodes = nodes1

        return tree


class ExternalReferences(ClientFunctionSet):
    '''
    API functionality for external references from a query sequence.

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        xrefs = c.xrefs('AAA')
    '''
    def __call__(self, sequence):
        '''
        Retrieve external references for a particular sequence.

        :param str sequence: query sequence or pattern
        :param bool as_dataframe: whether to return as pandas data frame, optional

        :return: list of cross references and match information
        :rtype: ClientResponse
        '''
        return self._client._request(action='xref',
                                     params={'search': sequence})


class PairwiseRelations(ClientFunctionSet):
    '''
    API functionality for pairwise relations..

    Access indirectly, via the client.

    Example::

        from omadb import Client
        c = Client()
        arath_wheat_pairs = list(c.pairwise('ARATH', 'WHEAT', progress=True))
    '''
    def __call__(self, genome_id1, genome_id2, chr1=None, chr2=None,
                 rel_type=None, progress=False):
        '''
        List the pairwise relations among two genomes. 

        If genome_id1 == genome_id2, relations are close paralogues and
        homoeologues. If different, the relations are orthologues.

        By using the paramaters `chr1` and `chr2`, it is possible to limit the
        relations to a certain chromosome for one or both genomes. The ID of
        the chromosome corresponds to the IDs in, for example::

            from omadb import Client
            c = Client()
            r = c.genomes.genome('HUMAN')
            human_inparalogues = list(c.pairwise('HUMAN',
                                                 'HUMAN',
                                                 chr1=r.chromosomes[0].id,
                                                 chr2=r.chromosomes[3].id,
                                                 progress=True))

        :param genome_id1: unique identifier for first genome - either NCBI taxonomic identifier or UniProt species code.
        :type genome_id1: int or str
        :param genome_id2: unique identifier for second genome - either NCBI taxonomic identifier or UniProt species code.
        :type genome_id2: int or str
        :param chr1: ID of chromosome of interest in first genome
        :type chr1: str or None
        :param chr2: ID of chromosome of interest in second genome :type chr2: str or None
        :param rel_type: relationship type ('1:1', '1:many', 'many:1', or 'many:many'), optional
        :param rel_type: str or None
        :param bool progress: whether to show a progress bar during load (default False)

        :return: generator of pairwise relations.
        :rtype: ClientPagedResponse
        '''
        return self._client._request(action=['pairs', genome_id2],
                                     subject=genome_id1,
                                     params={'chr1': chr1,
                                             'chr2': chr2,
                                             'rel_type': rel_type},
                                     paginated=True,
                                     progress_desc=('Loading pairs'
                                                  if progress else None))
