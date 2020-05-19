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

"""Stores various configuration options and constants for Oppia-ml."""

import os

# The platform for the storage backend. This is used in the model-switching
# code in core/platform.
PLATFORM = 'gce'

# Whether we should serve the development or production experience.
# DEV_MODE should only be changed to False in the production environment.
DEV_MODE = True

# Default communication URL to be used to communicate with Oppia server.
# This URL will be used in local development.
DEFAULT_COMMUNICATION_URL = 'http://localhost'

# Default communication PORT to be used to communicate with Oppia server.
# This URL will be used in local development.
DEFAULT_COMMUNICATION_PORT = 8181

# Communication URL which is to be used in production environment to
# communicate with Oppia server.
SERVER_COMMUNICATION_URL = 'https://www.oppia.org'

# Communication PORT which is to be used in production environment to
# communicate with Oppia server.
SERVER_COMMUNICATION_PORT = 443

# Default ID which is to be used in local development environment.
DEFAULT_VM_ID = 'vm_default'

# Default shared key which is to be used in local development environment.
DEFAULT_VM_SHARED_SECRET = '1a2b3c4e'

# Name of metadata parameter which will contain ID of the VM in production
# environment in GCE.
METADATA_VM_ID_PARAM_NAME = 'vm_id'

# Name of metadata parameter which will contain shared secret of the VM in
# production environment in GCE.
METADATA_SHARED_SECRET_PARAM_NAME = 'shared_secret_key'

# Handler URL of Oppia which is used to retrieve jobs.
FETCH_NEXT_JOB_REQUEST_HANDLER = 'ml/nextjobhandler'

# Handler URL of Oppia which is used to store job result.
STORE_TRAINED_CLASSIFIER_MODEL_HANDLER = 'ml/trainedclassifierhandler'

# Algorithm IDs of different classifier algorithms. These IDs are used to obtain
# instance of classifier algorithm using algorithm_registry.
# Note: we need same IDs in Oppia as well.
ALGORITHM_IDS = ['TextClassifier']

# Path of the directory which stores classifiers.
CLASSIFIERS_DIR = os.path.join('core', 'classifiers')

# Path of directory which stores datasets for testing.
DATASETS_DIR = os.path.join('core', 'tests', 'datasets')

# Path of directory which stores pretrained classifier models for each
# classifier.
PRETRAINED_MODELS_PATH = os.path.join('core', 'tests', 'models')

# Wait for fixed amount of time when there are no pending job requests.
FIXED_TIME_WAITING = 'fixed_time_wait'

# Seconds to wait in case of fixed time waiting approach.
FIXED_TIME_WAITING_PERIOD = 60

# Default waiting method to be used when there are no pending job requests.
DEFAULT_WAITING_METHOD = FIXED_TIME_WAITING

# Prefix for data sent from Oppia to the Oppia-ml via JSON.
XSSI_PREFIX = ')]}\'\n'

# The regular expression used to identify whether a string contains float value.
# The regex must match with regex that is stored in feconf.py file of Oppia.
# If this regex needs to be modified then first of all shutdown Oppia-ml VM.
# Then update the regex constant in here and Oppia both.
# Run any migration job that is required to migrate existing trained models
# before starting Oppia-ml again.
FLOAT_VERIFIER_REGEX = (
    '^([-+]?\\d*\\.\\d+)$|^([-+]?(\\d*\\.?\\d+|\\d+\\.?\\d*)e[-+]?\\d*)$')
