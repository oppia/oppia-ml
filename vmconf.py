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

# The platform for the storage backend. This is used in the model-switching
# code in core/platform.
PLATFORM = 'gce'

# Whether we should serve the development or production experience.
# DEV_MODE should only be changed to False in the production environment.
DEV_MODE = True

# Default communication URL to be used to communicate with Oppia server.
# This URL will be used in local development.
DEFAULT_COMMUNICATION_URL = 'https://localhost'

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
FETCH_NEXT_JOB_REQUEST_HANDLER = 'oppia-ml/fetch/job-request'

# Handler URL of Oppia which is used to store job result.
STORE_TRAINED_CLASSIFIER_MODEL_HANDLER = 'oppia-ml/store/trained-classifier'
