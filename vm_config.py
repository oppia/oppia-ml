# Copyright 2017 The Oppia Authors. All Rights Reserved.
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

import logging
import os
import sys

# Root path of the repo.
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MANIFEST_FILE_PATH = os.path.join(ROOT_PATH, 'manifest.txt')

# Third-party library paths.
with open(MANIFEST_FILE_PATH, 'r') as f:
    THIRD_PARTY_LIB_PATHS = [
        os.path.join(
            ROOT_PATH, 'third_party',
            '%s-%s' % (line.split()[0], line.split()[1])
        ) for line in [x.strip() for x in f.readlines()]
        if line and not line.startswith('#')
    ]

def configure():
    """This function configures python environment."""
    log_format = ('%(levelname)s\t%(asctime)s %(module)s:%(lineno)d]'
                  ' %(name)s: "%(message)s"')
    logging.basicConfig(format=log_format, level=logging.INFO)
    try:
        _fix_third_party_lib_paths()
    except Exception as e:  # pylint: disable=broad-except
        logging.info('Failed to configure VM: \'%s\'', e.message)


def _fix_third_party_lib_paths():
    """Fixes third party libraries path in python environment."""
    for lib_path in THIRD_PARTY_LIB_PATHS:
        if not os.path.isdir(lib_path):
            raise Exception(
                'Invalid path for third_party library: %s' % lib_path)
        sys.path.insert(0, lib_path)
