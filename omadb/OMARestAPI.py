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
from pprint import pformat
from property_manager import lazy_property
from requests_cache.core import CachedSession
from urllib.request import quote as uri_quote
from tqdm import tqdm
import appdirs
import json
import numbers
import os
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
            if type(v) == dict:
                self.__dictionary__[k] = self._recurse_create(v)
            elif type(v) == list:
                self.__dictionary__[k] = [(x if type(x) != dict else
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


class ClientResponse(AttrDict):
    def __init__(self, response, client=None, _is_paginated=None):
        self.client = client
        super().__init__(response)

    def _recurse_create(self, *args, **kwargs):
        return type(self)(*args, client=self.client, **kwargs)

    def __getattr__(self, attr):
        r = super().__getattr__(attr)
        return r if (type(r) is not ClientRequest) else r()

    def _setup_extra(self, k, v):
        if type(v) == str and v.startswith(self.client.endpoint):
            self.__dictionary__[k] = ClientRequest(self.client, v)


class ClientPagedResponse(object):
    def __init__(self, client, response, progress_desc=None):
        self.client = client
        self.response = response
        self.progress_desc = ('' if progress_desc is None else progress_desc)

    def __iter__(self):
        r = self.response
        x = json.loads(r.content)

        pbar = tqdm(desc=self.progress_desc,
                    unit=' entries',
                    total=int(r.headers.get('X-Total-Count', len(x))),
                    disable=(len(self.progress_desc) == 0))

        for e in map(lambda e: ClientResponse(e, client=self.client), x):
            yield e
            pbar.update()

        while 'next' in r.links:
            r = self.client._request(uri=r.links['next']['url'], raw=True)
            for e in map(lambda e: ClientResponse(e, client=self.client),
                         json.loads(r.content)):
                yield e
                pbar.update()
        pbar.close()


class ClientRequest(object):
    def __init__(self, client, uri):
        self.client = client
        self.uri = uri

    def __call__(self):
        return self.client._request(uri=self.uri)

    def __str__(self):
        return self.uri

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
            if action is None or (type(action) is list and len(action) == 0):
                raise Exception('No action declared.')

            if type(action) is list:
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
                if type(content) is list:
                    return list(map(lambda x: ClientResponse(x, client=self),
                                    content))

                else:
                    return ClientResponse(content, client=self)
        else:
            if r.status_code in {400, 404}:
                content = json.loads(r.content)
                z = ClientResponse(content, client=self)
                if set(z.keys()) == {'detail'}:
                    response_status += '["' + z.detail + '"]'
            raise ClientException(response_status)

    def _is_paginated(self, r):
        return len(set(r.links.keys()) & {'next', 'last'}) == 2



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
        return 'HOG:{:07d}'.format(hog_id) if type(hog_id) is int else hog_id

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

    def members(self, hog_id, level=None):
        '''
        Retrieve list of protein entries in a given HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int
        :param str level: level of interest

        :return: list of members
        :rtype: list
        '''
        return self._client._request(action=['hog', 'members'],
                                     subject=self._ensure_hog_id(hog_id),
                                     params={'level': level},
                                     level=level).members

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

    def xrefs(self, hog_id, type=None):
        '''
        Retrieve external references for all members of a particular HOG.

        :param hog_id: unique identifier for a HOG, either HOG ID or one of its member proteins
        :type hog_id: str or int
        :param str level: level of interest

        :return: external references
        :rtype: dict
        '''
        return {m['omaid']: self._client.entries.xrefs(m['entry_nr'], type)
                for m in self.members(hog_id, level=level)}


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

    def gene_ontology(self, entry_id):
        '''
        Retrieve any associations to Gene Ontology terms for a protein.

        :param entry_id: a unique identifier for a protein
        :type entry_id: str or int

        :return: gene ontology associations
        :rtype: ClientResponse
        '''
        return self._client._request(action=['protein', 'ontology'],
                                     subject=entry_id)

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
    def __call__(self, seq):
        '''
        Annotate a sequence with GO terms based on annotations stored in the
        OMA database.

        :param str query: query sequence

        :return: results of fast function prediction
        :rtype: ClientResponse
        '''
        return self._client._request(action='function',
                                     params={'query': seq})


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

    def close_groups(self, group_id):
        '''
        Retrieve the sorted list of closely related groups for a given OMA
        group.

        :param group_id: unique identifier of a group - either group number, fingerprint or entry ID of a member.
        :type group_id: int or str

        :return: sorted list of closely related groups
        :rtype: list
        '''
        return self._client._request(action=['group', 'close_groups'],
                                     subject=group_id)


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

        :return: list of cross references
        :rtype: list
        '''
        return self._client._request(action='xref', params={'search': sequence})


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
        :param chr2: ID of chromosome of interest in second genome
        :type chr2: str or None
        :param rel_type: relationship type ('1:1', '1:many', 'many:1', or 'many:many'), optional
        :param rel_type: str or None

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
