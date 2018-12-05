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
import dendropy
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

        pbar = tqdm(desc=self.progress_desc,
                    unit=' entries',
                    total=int(r.headers['X-Total-Count']),
                    disable=(len(self.progress_desc) == 0))

        for e in map(lambda e: ClientResponse(e, client=self.client),
                     json.loads(r.content)):
            yield e
            pbar.update()

        while 'next' in r.links:
            r = self.client.request(uri=r.links['next']['url'], raw=True)
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
        return self.client.request(uri=self.uri)

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
    HEADERS = {'Content-type': 'application/json',
               'Accept': 'application/json'}
    TIMEOUT = 60
    PER_PAGE = 10000
    RAMCACHE_SIZE = 10000

    def __init__(self, endpoint='omabrowser.org/api', persistent_cached=False,
                 persistent_cache_path=None):
        self.endpoint = ('https://' + endpoint
                         if not endpoint.startswith('http')
                         else endpoint)
        if persistent_cached:
            if persistent_cache_path is None:
                self.CACHE_PATH = appdirs.user_cache_dir('py' + __package__)
            else:
                self.CACHE_PATH = os.path.abspath(persistent_cache_path)
            self._version_check()
            self._setup_cache()
        self._setup()

    def clear_cache(self, restart=None):
        if hasattr(self, 'session'):
            self.session.close()
            del self.session
            shutil.rmtree(self.CACHE_PATH)
        if restart is True:
            self._version_check()
            self._setup_cache()

    def _setup(self):
        self.genomes = Genomes(self)
        self.entries = self.proteins = Entries(self)
        self.hogs = HOGs(self)
        self.groups = OMAGroups(self)
        self.function = Function(self)
        self.taxonomy = Taxonomy(self)
        self.pairwise = PairwiseRelations(self)

    def _setup_cache(self):
        os.makedirs(self.CACHE_PATH, exist_ok=True)
        self.session = CachedSession(cache_name=os.path.join(self.CACHE_PATH,
                                                             'api-cache'),
                                     backend='sqlite')

    @lazy_property
    def version(self):
        return self.request(action='version')

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

    def request(self, request_type='get', **kwargs):
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

            elif self._is_paginated(r):
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
            if r.status_code == 404:
                content = json.loads(r.content)
                z = ClientResponse(content, client=self)
                if set(z.keys()) == {'detail'}:
                    response_status += ' -- ' + z.detail
            raise ClientException(response_status)

    def _is_paginated(self, r):
        return len(set(r.links.keys()) & {'next', 'last'}) == 2


class ClientFunctionSet(object):
    def __init__(self, client):
        self._client = client
        if hasattr(self, '_setup'):
            self._setup()


class Genomes(ClientFunctionSet):
    def __getitem__(self, x):
        return self.genome(x)

    def __iter__(self):
        yield from self.genomes.items()

    @property
    def list(self):
        return self.genomes

    @lazy_property
    def genomes(self):
        r = {sp.pop('code'): sp
             for sp in self._client.request(action='genome',
                                            paginated=True)}
        return ClientResponse(r, client=self._client)

    def genome(self, sp_code):
        return self._client.request(action='genome', subject=sp_code)

    def proteins(self, sp_code, progress=False):
        desc = 'Retrieving proteins for {}'.format(sp_code) if progress else ''
        yield from self._client.request(action=['genome', 'proteins'],
                                        subject=sp_code,
                                        paginated=True,
                                        progress_desc=desc)


class HOGs(ClientFunctionSet):
    def _ensure_hog_id(self, hog_id):
        return 'HOG:{:07d}'.format(hog_id) if type(hog_id) is int else hog_id

    def __getitem__(self, x):
        return self.info(x)

    def at_level(self, level):
        yield from self._client.request(action='hog', params={'level': level})

    def info(self, hog_id):
        return self._client.request(action='hog',
                                    subject=self._ensure_hog_id(hog_id))

    def members(self, hog_id, level=None):
        # NOTE: Currently only support root-level
        return self._client.request(action=['hog', 'members'],
                                    subject=self._ensure_hog_id(hog_id),
                                    level=level).members

    def xrefs(self, hog_id, xref_type=None):
        return {m['omaid']: self._client.entries.xrefs(m['entry_nr'], xref_type)
                for m in self.members(hog_id)}


class Entries(ClientFunctionSet):
    def __getitem__(self, x):
        return self.info(x)

    def _setup(self):
        self.orthologs = self.orthologues
        self.search = SequenceSearch(self._client)

    def info(self, entry_id):
        if isinstance(entry_id, numbers.Number) or isinstance(entry_id, str):
            return self._client.request(action='protein', subject=entry_id)
        else:
            return self._client.request(action='protein',
                                        subject='bulk_retrieve',
                                        data={'ids': list(entry_id)},
                                        request_type='post')

    def domains(self, entry_id):
        return self._client.request(action=['protein', 'domains'],
                                    subject=entry_id)

    def gene_ontology(self, entry_id):
        return self._client.request(action=['protein', 'ontology'],
                                    subject=entry_id)

    def homoeologs(self, entry_id):
        return self.homoeologues(entry_id)

    def homoeologues(self, entry_id):
        return self._client.request(action=['protein', 'homoeologs'],
                                    subject=entry_id)

    def orthologues(self, entry_id, rel_type=None):
        return self._client.request(action=['protein', 'orthologs'],
                                    subject=entry_id,
                                    params=({'rel_type': rel_type}
                                            if rel_type is not None else None))

    def xrefs(self, entry_id, xref_type=None):
        def reformat(r):
            z = defaultdict(list)
            for x in r:
                z[x['source']].append(x['xref'])
            return z

        r = self._client.request(action=['protein', 'xref'],
                                 subject=entry_id)
        if xref_type is not None:
            return set(map(lambda x: x['xref'],
                           filter(lambda x: x['source'] == xref_type, r)))
        else:
            return reformat(r)


class SequenceSearch(ClientFunctionSet):
    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

    def search(self, sequence, search=None, full_length=None):
        params = {'query': sequence,
                  'full_length': bool(full_length)}

        search_valid = {'exact', 'approximate', 'mixed'}
        search = search.lower() if search is not None else None
        if (search not in search_valid and search is not None):
            raise ValueError('{} is not a valid search method. Choose from {}'
                             .format(search, ', '.join(search_valid)))
        elif search is not None:
            params['search'] = search

        return self._client.request(action='sequence', params=params)


class Function(ClientFunctionSet):
    def __call__(self, seq):
        return self._client.request(action='function',
                                    params={'query': seq})


class OMAGroups(ClientFunctionSet):
    def __getitem__(self, x):
        return self.info(x)

    @property
    def list(self):
        return self.groups

    @lazy_property
    def groups(self):
        # TODO: work out how to iter as well as store?
        # Could this be an async dict?
        return {r['oma_group']: r['group_url']
                for r in self._client.request(action='group')}

    def info(self, group_id):
        return self._client.request(action='group', subject=group_id)

    def close_groups(self, group_id):
        return self._client.request(action=['group', 'close_groups'],
                                    subject=group_id)


class Taxonomy(ClientFunctionSet):
    def __call__(self, *args, **kwargs):
        return self.tree(*args, **kwargs)

    def get(self, members=None):
        return self._client.request(action='taxonomy',
                                    params=({'members': ','.join(members)}
                                            if members is not None else None))

    def read(self, root):
        return self._client.request(action=['taxonomy', root])

    def tree(self, members=None, root=None, with_names=None):
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


class ExternalReferences(ClientFunctionSet):
    def __call__(self, *args, **kwargs):
        return self.list(*args, **kwargs)

    def list(self, pattern):
        return self._client.request(action='xref', params={'search': pattern})


class PairwiseRelations(ClientFunctionSet):
    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def read(self, genome_id1, genome_id2, chr1=None, chr2=None, rel_type=None,
             progress=False):
        return self._client.request(action=['pairs', genome_id2],
                                    subject=genome_id1,
                                    chr1=chr1,
                                    chr2=chr2,
                                    rel_type=rel_type,
                                    progress_desc=('Loading pairs'
                                                  if progress else None))
