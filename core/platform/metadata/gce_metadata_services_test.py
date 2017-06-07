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

"""Tests for gce metadata services"""

from core.platform.metadata import gce_metadata_services
from core.tests import test_utils

class MetadataServicesTests(test_utils.GenericTestBase):
    """Tests for metadata accessing methods."""

    def test_that_metadata_is_fetched(self):
        """Test that metadata param is fetched correctly."""
        # Get value of 'param' parameter.
        param_name = 'param'
        metadata_url = '%s/%s' % (
            gce_metadata_services.METADATA_ATTRIBUTES_URL, param_name)
        metadata_headers = gce_metadata_services.METADATA_HEADERS
        metadata_value = 'value'

        with self.put_get_request(
            metadata_url, metadata_value, 200, metadata_headers):
            resp = gce_metadata_services.get_metadata_param(param_name)
            self.assertEqual(resp.json(), 'value')
