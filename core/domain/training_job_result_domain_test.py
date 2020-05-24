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

"""Tests for gce metadata services"""

from core.domain import training_job_result_domain
from core.domain.protofiles import text_classifier_pb2
from core.tests import test_utils

class TrainingJobResultTests(test_utils.GenericTestBase):
    """Tests for TrainingJobResult domain object."""

    def test_that_validation_checks_are_correct(self):
        """Ensure that validation checks work as expected."""
        job_id = 'job_id'
        algorithm_id = 'TextClassifier'
        classifier_data = text_classifier_pb2.TextClassifier()
        classifier_data.model_json = 'dummy model'
        job_result = training_job_result_domain.TrainingJobResult(
            job_id, algorithm_id, classifier_data)
        job_result.validate()

        classifier_data = 'simple classifier'
        job_result = training_job_result_domain.TrainingJobResult(
            job_id, algorithm_id, classifier_data)
        with self.assertRaisesRegexp(
            Exception,
            'Expected classifier data of type TextClassifier'):
            job_result.validate()