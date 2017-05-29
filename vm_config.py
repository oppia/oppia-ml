# Copyright 2016 The Oppia Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for configuration of python environment."""

import os
import sys

# Root path of the repo.
ROOT_PATH = os.path.dirname(__file__)

# Third party libraris path.
THIRD_PARTY_LIBS = [
    os.path.join(ROOT_PATH, 'third_party', 'sklearn-0.18.1'),
    os.path.join(ROOT_PATH, 'third_party', 'numpy-1.12.1'),
    os.path.join(ROOT_PATH, 'third_party', 'scipy-0.19.0'),
]

def configure():
    """This function configures python environment."""
    _fix_third_party_libs_path()


def _fix_third_party_libs_path():
    """Fixes third party libraries path in python environment."""
    for lib_path in THIRD_PARTY_LIBS:
        if not os.path.isdir(lib_path):
            raise Exception(
                'Invalid path for third_party library: %s' % lib_path)
        sys.path.insert(0, lib_path)
