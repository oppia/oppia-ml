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

from core.domain import remote_access_services
from core.domain import training_job_result_domain
from core.domain.proto import text_classifier_pb2
from core.domain.proto import training_job_response_payload_pb2
from core.tests import test_utils
import vmconf

class RemoteAccessServicesTests(test_utils.GenericTestBase):
    """Tests for remote accessing methods."""

    def test_that_generate_signature_works_correctly(self):
        """Test that generate signature function is working as expected."""
        with self.swap(vmconf, 'DEFAULT_VM_SHARED_SECRET', '1a2b3c4e'):
            message = 'vm_default'
            vm_id = 'vm_default'
            signature = remote_access_services.generate_signature(
                message.encode(), vm_id.encode())

        expected_signature = (
            '740ed25befc87674a82844db7769436edb7d21c29d1c9cc87d7a1f3fdefe3610')
        self.assertEqual(signature, expected_signature)
        self.assertEqual(signature, "hi")

    def test_next_job_gets_fetched(self):
        """Test that next job is fetched correctly."""
        # Callback for post request.
        @self.callback
        def post_callback(request):
            """Callback for post request."""
            self.assertIn('vm_id', request.payload.keys())
            self.assertIn('message', request.payload.keys())
            self.assertIn('signature', request.payload.keys())
            job_data = {
                'job_id': '1',
                'algorithm_id': 'ab',
                'training_data': {},
                'algorithm_version': 1
            }
            return job_data

        with self.set_job_request_post_callback(post_callback):
            resp = remote_access_services.fetch_next_job_request()

        self.assertIn('job_id', resp.keys())
        self.assertIn('algorithm_id', resp.keys())
        self.assertIn('training_data', resp.keys())
        self.assertIn('algorithm_version', resp.keys())

        self.assertEqual(resp['job_id'], '1')
        self.assertEqual(resp['algorithm_id'], 'ab')
        self.assertEqual(resp['algorithm_version'], 1)
        self.assertDictEqual(resp['training_data'], {})

    def test_result_gets_stored_correctly(self):
        """Test that correct results are stored."""

        frozen_model_proto = text_classifier_pb2.TextClassifierFrozenModel()
        frozen_model_proto.model_json = 'model_json'
        algorithm_id = 'TextClassifier'
        job_id = '123'
        job_result = training_job_result_domain.TrainingJobResult(
            job_id, algorithm_id, frozen_model_proto)

        # Callback for post request.
        @self.callback
        def post_callback(request):
            """Callback for post request."""
            payload = (
                training_job_response_payload_pb2.TrainingJobResponsePayload())
            payload.ParseFromString(request.body)
            self.assertEqual(payload.job_result.job_id, '123')
            self.assertEqual(
                payload.job_result.WhichOneof('classifier_frozen_model'),
                'text_classifier')
            self.assertEqual(
                payload.job_result.text_classifier.model_json, 'model_json')

        with self.set_job_result_post_callback(post_callback):
            status = remote_access_services.store_trained_classifier_model(
                job_result)
        self.assertEqual(status, 200)

    def test_that_job_result_is_validated_before_storing(self):
        """Ensure that JobResult domain object is validated before proceeding
        to store it.
        """
        frozen_model_proto = text_classifier_pb2.TextClassifierFrozenModel()
        frozen_model_proto.model_json = 'model_json'
        algorithm_id = 'TextClassifier'
        job_id = '123'

        job_result = training_job_result_domain.TrainingJobResult(
            job_id, algorithm_id, frozen_model_proto)

        check_valid_call = {'validate_has_been_called': False}
        def validate_check():
            """Assert that validate function is called."""
            check_valid_call['validate_has_been_called'] = True

        @self.callback
        def post_callback(request): # pylint: disable=unused-argument
            """Callback for post request."""
            return

        with self.set_job_result_post_callback(post_callback):
            with self.swap(job_result, 'validate', validate_check):
                remote_access_services.store_trained_classifier_model(
                    job_result)
        self.assertTrue(check_valid_call['validate_has_been_called'])
