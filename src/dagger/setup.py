# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""catkin Python package declaration.

Used with ``catkin_python_setup()`` in ``CMakeLists.txt`` so the Python
packages under ``src/dagger/dagger`` are installed into catkin ``devel`` and
``install`` spaces. Downstream nodes import them via ``from dagger...``.
"""

from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        'dagger',
        'dagger.infra',
        'dagger.hardware',
        'dagger.collectors',
        'dagger.reward_models',
    ], )

setup(**setup_args)
