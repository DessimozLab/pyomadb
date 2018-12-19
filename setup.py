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
from setuptools import setup, find_packages

name = 'omadb'
with open('{:s}/__init__.py'.format(name), 'rt') as fp:
    for line in fp:
        if line.startswith('__version__'):
            exec(line.rstrip())

requirements = ['pprint', 'property_manager', 'requests_cache', 'tqdm',
                'appdirs', 'requests', 'pandas']

desc = 'Client to the OMA browser, using the REST API.'

setup(
    name=name,
    version=__version__,
    author='Alex Warwick Vesztrocy',
    author_email='alex@warwickvesztrocy.co.uk',
    url='https://github.com/DessimozLab/pyomadb',
    description=desc,
    long_description=desc + '\n Documentation available `here <https://dessimozlab.github.io/pyomadb/build/html/>`_.',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={'dendropy': ['dendropy'],
                    'ete3': ['ete3'],
                    'goea': ['goatools'],
                    'hog_analysis': ['pyham']},
    python_requires=">=3.6",
    license='LGPLv3')
