# PyOMADB

PyOMADB is a python client to the OMA browser, using the REST API. As such, it requires a stable internet connection to operate. We also provide a _similar wrapper for R (https://github.com/DessimozLab/omadb)_.

Documentation is available <a href="http://dessimozlab.github.io/pyomadb/build/html/">here</a>. A notebook containing examples of how to use the package is available <a href="https://github.com/DessimozLab/pyomadb/blob/master/examples/pyomadb-examples.ipynb">here</a>.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DessimozLab/pyomadb/master?filepath=examples%2Fpyomadb-examples.ipynb)

<b>New (version 2.1.0)</b> - in order to facilitate the functional and evolutionary aspects of coronaviruses, the new <a href="https://corona.omabrowser.org/">Corona OMA Browser</a> has been launched. Please see the [Corona-OMA Example](#corona-oma-example) for more information.


## Citation
If you use our package in your work, please consider citing:

_Kaleb K, Warwick Vesztrocy A, Altenhoff A and Dessimoz C. Expanding the Orthologous Matrix (OMA) programmatic interfaces: REST API and the OmaDB packages for R and Python. F1000Research 2019, 8:42
(https://doi.org/10.12688/f1000research.17548.1)_

## Installation

The package requires Python 3 (>=3.6). The easiest way to install is using `pip`, to install the <a href="https://pypi.org/project/omadb/">package from PyPI</a>.

```
pip install omadb
```

## Documentation

Documentation is available <a href="http://dessimozlab.github.io/pyomadb/build/html/">here</a>.

## Example

As an example, if a user has the ID of their gene of interest, they can find the corresponding OMA entry as follows:

```
from omadb import Client
c = Client()

prot_id = 'P53_RAT'
r = c.proteins[prot_id]  # Can also be called as c.proteins.info(prot_id)
```

This is then a Python dictionary containing information about this entry. Some information is lazily loaded, for example:

```
orth = r.orthologs  # Will lazily load in the background.
```

Alternatively, if the user has the sequence but no ID, the closest entry in OMA can be identified as so:

```
seq = 'MKLVFLVLLFLGALGLCLAGRRRSVQWCAVSQPEATKCFQWQRNMRKVRGPPVSCIKRDSPIQCIQAIAENRADAVTLDGGFIYEAGLAPYKLRPVAAEVYGTERQPRTHYYAVAVVKKGGSFQLNELQGLKSCHTGLRRTAGWNVPIGTLRPFLNWTGPPEPIEAAVARFFSASCVPGADKGQFPNLCRLCAGTGENKCAFSSQEPYFSYSGAFKCLRDGAGDVAFIRESTVFEDLSDEAERDEYELLCPDNTRKPVDKFKDCHLARVPSHAVVARSVNGKEDAIWNLLRQAQEKFGKDKSPKFQLFGSPSGQKDLLFKDSAIGFSRVPPRIDSGLYLGSGYFTAIQNLRKSEEEVAARRARVVWCAVGEQELRKCNQWSGLSEGSVTCSSASTTEDCIALVLKGEADAMSLDGGYVYTAGKCGLVPVLAENYKSQQSSDPDPNCVDRPVEGYLAVAVVRRSDTSLTWNSVKGKKSCHTAVDRTAGWNIPMGLLFNQTGSCKFDEYFSQSCAPGSDPRSNLCALCIGDEQGENKCVPNSNERYYGYTGAFRCLAENAGDVAFVKDVTVLQNTDGNNNEAWAKDLKLADFALLCLDGKRKPVTEARSCHLAMAPNHAVVSRMDKVERLKQVLLHQQAKFGRNGSDCPDKFCLFQSETKNLLFNDNTECLARLHGKTTYEKYLGPQYVAGITNLKKCSTSPLLEACEFLRK'

r = c.proteins.search(seq)
```

For further examples that correspond to the `R` versions given in the paper, see the <a href="https://github.com/DessimozLab/pyomadb/blob/master/examples/pyomadb-examples.ipynb">Jupyter notebook</a>, which is also available on <a href="https://mybinder.org/v2/gh/DessimozLab/pyomadb/master?filepath=examples%2Fpyomadb-examples.ipynb">mybinder</a>.

## Corona-OMA Example

In order to facilitate the functional and evolutionary aspects of coronaviruses, the new <a href="https://corona.omabrowser.org/">Corona OMA Browser</a> has been launched.

The endpoint for the Corona OMA Browser is `corona.omabrowser.org/api` and can bused in PyOMADB by importing `CoronaClient` instead of `Client`.

```
from omadb import CoronaClient
c = CoronaClient()       # Connects to Corona OMA endpoint

prot_id = 'R1AB_SARS2'   # Severe acute respiratory syndrome coronavirus 2
r = c.proteins[prot_id]  # Can also be called as c.proteins.info(prot_id)
```

This is then a Python dictionary containing information about this entry. Some information is lazily loaded, for example:

```
orth = r.orthologs  # Will lazily load in the background.
```

## License

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
