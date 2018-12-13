PyOMADB
=============

PyOMADB is intended as a user-friendly wrapper around the OMA REST API.

Client
++++++

.. autoclass:: omadb.Client

  .. automethod:: __init__
  
  .. attribute:: genomes

  Instance of :py:class:`omadb.OMARestAPI.Genomes`.

  .. attribute:: entries

  Instance of :py:class:`omadb.OMARestAPI.Entries`.

  .. attribute:: proteins

  Synonym of `entries`.

  .. attribute:: hogs

  Instance of :py:class:`omadb.OMARestAPI.HOGs`.

  .. attribute:: groups

  Instance of :py:class:`omadb.OMARestAPI.OMAGroups`.

  .. attribute:: function

  Instance of :py:class:`omadb.OMARestAPI.Function`.

  .. attribute:: taxonomy

  Instance of :py:class:`omadb.OMARestAPI.Taxonomy`.

  .. attribute:: pairwise

  Instance of :py:class:`omadb.OMARestAPI.PairwiseRelations`.

  .. attribute:: xrefs

  Instance of :py:class:`omadb.OMARestAPI.ExternalReferences`.

  .. attribute:: external_references

  Synonym of `xrefs`.

  .. automethod:: omadb.Client.clear_cache

Genomes
+++++++
.. autoclass:: omadb.OMARestAPI.Genomes
  :members:

  .. automethod:: __getitem__

  .. automethod:: __iter__

Entries
+++++++
.. autoclass:: omadb.OMARestAPI.Entries
  :members:

  .. automethod:: __getitem__

HOGs
++++
.. autoclass:: omadb.OMARestAPI.HOGs
  :members:

  .. automethod:: __getitem__

OMAGroups
+++++++++
.. autoclass:: omadb.OMARestAPI.OMAGroups
  :members:

  .. automethod:: __getitem__

  .. automethod:: __iter__

Function
++++++++
.. autoclass:: omadb.OMARestAPI.Function
  :members:

  .. automethod:: __call__

Taxonomy
++++++++
.. autoclass:: omadb.OMARestAPI.Taxonomy
  :members:

PairwiseRelations
+++++++++++++++++
.. autoclass:: omadb.OMARestAPI.PairwiseRelations
  :members:

  .. automethod:: __call__

ExternalReferences
++++++++++++++++++
.. autoclass:: omadb.OMARestAPI.ExternalReferences
  :members:

  .. automethod:: __call__

ClientResponse
++++++++++++++
.. autoclass:: omadb.OMARestAPI.ClientResponse

AttrDict
++++++++
.. autoclass:: omadb.OMARestAPI.AttrDict

ClientPagedResponse
+++++++++++++++++++
.. autoclass:: omadb.OMARestAPI.ClientPagedResponse


License
+++++++

PyOMADB is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyOMADB is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with PyOMADB. If not, see <http://www.gnu.org/licenses/>.
