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

"""Tests for remote access services."""

import json

from core.services import remote_access_services
from core.tests import test_utils
import vmconf

class RemoteAccessServicesTests(test_utils.GenericTestBase):
    """Tests for remote accessing methods."""

    def test_that_generate_signature_works_correctly(self):
        """Test that generate signature function is working as expected."""
        data = {
            'vm_id': 'vm_default'
        }

        with self.swap(vmconf, 'DEFAULT_VM_SHARED_SECRET', '1a2b3c4e'):
            sign = remote_access_services.generate_signature(data)

        expected_sign = (
            '45a6a6200a11d13d56ad5c505005e294f675f8c79943e5afd14b922e2f7a287d')
        self.assertEqual(sign, expected_sign)

    def test_next_job_gets_fetched(self):
        """Test that next job is fetched correctly."""
        with self.save_new_job_request('1', 'ab', {}):
            resp = remote_access_services.fetch_next_job_request()

        self.assertIn('job_id', resp.keys())
        self.assertIn('algorithm_id', resp.keys())
        self.assertIn('training_data', resp.keys())

        self.assertEqual(resp['job_id'], '1')
        self.assertEqual(resp['algorithm_id'], 'ab')
        self.assertDictEqual(resp['training_data'], {})

    def test_result_gets_stored_correctly(self):
        """Test that correct results are stored."""

        job_result_dict = {
            'job_id': '123',
            'classifier_data': {
                'param': 'val'
            }
        }

        # Callback for post request.
        @self.callback
        def post_callback(request):
            """Callback for post request."""
            payload = json.loads(request.body)
            self.assertEqual(payload['job_id'], '123')
            classifier_data = {
                'param': 'val'
            }
            self.assertDictEqual(classifier_data, payload['classifier_data'])

        with self.set_job_result_post_callback(post_callback):
            resp = remote_access_services.store_trained_classifier_model(
                job_result_dict)
        self.assertEqual(resp.status_code, 200)

    def test_exception_is_raised_when_classifier_data_is_inappropriate(self):
        """Test that correct results are stored."""
        job_result_dict = 123

        with self.assertRaisesRegexp(
            Exception, 'job_result_dict must be in dict format'):
            remote_access_services.store_trained_classifier_model(
                job_result_dict)

        job_result_dict = {}

        with self.assertRaisesRegexp(
            Exception, 'job_result_dict must contain \'job_id\'.'):
            remote_access_services.store_trained_classifier_model(
                job_result_dict)

        job_result_dict = {
            'job_id': 'id'
        }

        with self.assertRaisesRegexp(
            Exception, 'job_result_dict must contain \'classifier_data\'.'):
            remote_access_services.store_trained_classifier_model(
                job_result_dict)
