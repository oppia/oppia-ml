# coding: utf-8
#
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

"""Functions for retrieving metadata of GCE."""

import requests


# Following URL is used to retrieve metadata of gce instance.
# For more information check following link:
# https://cloud.google.com/compute/docs/storing-retrieving-metadata
METADATA_ATTRIBUTES_URL = (
    'https://metadata.google.internal/computeMetadata/v1/instance/attributes/')


METADATA_HEADERS = {
    'Metadata-Flavor': 'Google'
}


def get_metadata_param(param_name):
    """Fetch value of metadata parameter.

    Args:
        param_name: str. Name of the parameter to be fetched.
    """

    # Send a request to get param details.
    value = requests.get(
        '%s/%s' % (METADATA_ATTRIBUTES_URL, param_name),
        headers=METADATA_HEADERS)

    return value
