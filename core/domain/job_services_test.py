# coding: utf-8
#
# Copyright 2020 The Oppia Authors. All Rights Reserved.
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
import os

from core.domain import job_services
from core.domain.proto import text_classifier_pb2
from core.domain.proto import training_job_response_payload_pb2
from core.tests import test_utils
import vmconf


def _load_training_data():
    file_path = os.path.join(
        vmconf.DATASETS_DIR, 'string_classifier_data.json')
    with open(file_path, 'r') as f:
        training_data = json.loads(f.read())
    return training_data


class JobServicesTests(test_utils.GenericTestBase):
    """Tests for job service function ."""

    def _get_post_callback_for_next_job(self, job_data):
        def post_callback(request):
            """Callback for post request."""
            self.assertIn('vm_id', request.payload.keys())
            self.assertIn('message', request.payload.keys())
            self.assertIn('signature', request.payload.keys())
            return job_data
        return self.callback(post_callback)

    def test_next_job_is_fetched_correctly(self):
        """Test that next job is fetched correctly."""
        job_data = {
            'job_id': '1',
            'algorithm_id': 'TextClassifier',
            'training_data': [],
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            job_data = job_services.get_next_job()

        self.assertIn('job_id', job_data.keys())
        self.assertIn('algorithm_id', job_data.keys())
        self.assertIn('training_data', job_data.keys())
        self.assertIn('algorithm_version', job_data.keys())

        self.assertEqual(job_data['job_id'], '1')
        self.assertEqual(job_data['algorithm_id'], 'TextClassifier')
        self.assertEqual(job_data['algorithm_version'], 1)
        self.assertEqual(job_data['training_data'], [])

    def test_job_validation_raises_exception_on_incorrect_job_data_type(self):
        """Test that job validation raises exception if job_id is missing
        in the received job data.
        """
        job_data = 'a simple job'
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'Invalid format of job data'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_missing_job_id(self):
        """Test that job validation raises exception if job_id is missing
        in the received job data.
        """
        job_data = {
            'algorithm_id': 'TextClassifier',
            'training_data': [],
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)
        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'job data should contain job id'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_missing_algorithm_id(self):
        """Test that job validation raises exception if algorithm_id is missing
        in the received job data.
        """
        job_data = {
            'job_id': '1',
            'training_data': [],
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)
        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'job data should contain algorithm id'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_missing_training_data(self):
        """Test that job validation raises exception if training_data is missing
        in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 'TextClassifier',
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)
        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'job data should contain training data'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_missing_algorithm_version(self):
        """Test that job validation raises exception if algorithm_version is
        missing in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 'TextClassifier',
            'training_data': []
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'job data should contain algorithm version'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_invalid_job_id(self):
        """Test that job validation raises exception if job_id is invalid
        in the received job data.
        """
        # Callback for post request.
        job_data = {
            'job_id': 123,
            'algorithm_id': 'TextClassifier',
            'training_data': [],
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'Expected job id to be unicode'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_invalid_algorithm_id(self):
        """Test that job validation raises exception if algorithm_id is invalid
        in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 123,
            'training_data': [],
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'Expected algorithm id to be unicode'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_invalid_training_data(self):
        """Test that job validation raises exception if training_data is invalid
        in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 'TextClassifier',
            'training_data': 'data',
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'Expected training data to be a list'):
                job_services.get_next_job()

        job_data['training_data'] = [{'answers': ['a', 'b', 'c']}]
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception,
                'Expected answer_group_index to be a key in training_data'):
                job_services.get_next_job()

        job_data['training_data'] = [{'answer_group_index': 1}]
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception,
                'Expected answers to be a key in training_data'):
                job_services.get_next_job()

        job_data['training_data'] = [{
            'answer_group_index': '1',
            'answers': ['a', 'b', 'c']
        }]
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception,
                'Expected answer_group_index to be an int'):
                job_services.get_next_job()

        job_data['training_data'] = [{
            'answer_group_index': 1,
            'answers': 'answer1'
        }]
        post_callback = self._get_post_callback_for_next_job(job_data)
        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception,
                'Expected answers to be a list'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_invalid_algorithm_version(self):
        """Test that job validation raises exception if algorithm_version is
        invalid in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 'TextClassifier',
            'training_data': [],
            'algorithm_version': '123'
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'Expected algorithm version to be integer'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_unknown_algorithm_id(self):
        """Test that job validation raises exception if algorithm_id is
        unknown in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 'ab',
            'training_data': [],
            'algorithm_version': 1
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception, 'Invalid algorithm id ab'):
                job_services.get_next_job()

    def test_job_validation_raises_exception_on_unknown_algorithm_version(self):
        """Test that job validation raises exception if algorithm_version is
        unknown in the received job data.
        """
        job_data = {
            'job_id': '1',
            'algorithm_id': 'TextClassifier',
            'training_data': [],
            'algorithm_version': 1000
        }
        post_callback = self._get_post_callback_for_next_job(job_data)

        with self.set_job_request_post_callback(post_callback):
            with self.assertRaisesRegexp(
                Exception,
                'Classifier version 1 mismatches algorithm version 1000 '
                'received in job data'):
                job_services.get_next_job()

    def test_result_gets_stored_correctly(self):
        """Test that correct results are stored."""

        frozen_model_proto = text_classifier_pb2.TextClassifierFrozenModel()
        frozen_model_proto.model_json = 'model_json'
        algorithm_id = 'TextClassifier'
        job_id = '123'

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
            status = job_services.store_job_result(
                job_id, algorithm_id, frozen_model_proto)
        self.assertEqual(status, 200)

    def test_train_classifier(self):
        """Ensure that train classifier trains classifier."""
        training_data = _load_training_data()
        frozen_model = job_services.train_classifier(
            'TextClassifier', 1, training_data)
        self.assertTrue(isinstance(
            frozen_model, text_classifier_pb2.TextClassifierFrozenModel))

    def test_train_classifier_returns_none_on_invalid_algorithm_versio(self):
        """Ensure that train classifier trains classifier."""
        training_data = _load_training_data()
        frozen_model = job_services.train_classifier(
            'TextClassifier', 1, training_data)
        self.assertTrue(isinstance(
            frozen_model, text_classifier_pb2.TextClassifierFrozenModel))
