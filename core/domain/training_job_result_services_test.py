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

"""Tests for training job results services."""

from core.classifiers import algorithm_registry
from core.domain.protofiles import text_classifier_pb2
from core.domain.protofiles import training_job_data_pb2
from core.domain import training_job_result_domain
from core.domain import training_job_result_services
from core.tests import test_utils

class TrainingJobResultServicesTests(test_utils.GenericTestBase):
    """Generic tests for functions in training job result services."""

    def test_that_all_algorithms_have_job_result_information(self):
        """Test that all algorithms have properties to identify name and type
        of attribute in job result proto which stores classifier data for that
        algorithm.
        """
        job_result_proto = training_job_data_pb2.TrainingJobData.JobResult()
        for classifier in algorithm_registry.Registry.get_all_classifiers():
            self.assertIsNotNone(classifier.name_in_job_result_proto)
            attribute_type_name = type(getattr(
                job_result_proto, classifier.name_in_job_result_proto)).__name__
            self.assertEqual(
                attribute_type_name, classifier.type_in_job_result_proto)

    def test_that_training_job_result_proto_is_correct(self):
        """Ensure that the JobResult proto is correctly generated from
        TrainingJobResult domain object.
        """
        classifier_data = text_classifier_pb2.TextClassifier()
        classifier_data.model_json = 'dummy model'
        job_id = 'job_id'
        algorithm_id = 'TextClassifier'
        classifier = algorithm_registry.Registry.get_classifier_by_algorithm_id(
            algorithm_id)
        job_result = training_job_result_domain.TrainingJobResult(
            job_id, algorithm_id, classifier_data)
        job_result_proto = (
            training_job_result_services.get_proto_message_from_training_job_result( # pylint: disable=line-too-long
                job_result))

        # Lint test for no-member needs to be disabled as protobuf generated
        # classes are metaclasses and hence their attributes are defined at
        # runtime.
        self.assertEqual(job_result_proto.job_id, job_id) # pylint: disable=no-member
        self.assertEqual(
            job_result_proto.WhichOneof('classifier_data'),
            classifier.name_in_job_result_proto)
