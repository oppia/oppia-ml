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

from core.platform.protobuf import gce_protobuf_services
from core.platform.protobuf.protofiles import text_classifier_pb2 # pylint: disable=no-name-in-module
from core.tests import test_utils
import vmconf

class TrainingJobDataResultTests(test_utils.GenericTestBase):
    """Tests for TrainingJobResultMessage class."""

    def test_that_all_algorithm_ids_have_mapping(self):
        """Test that TrainingJobDataMessage contains mapping for all
        algorithm ids.
        """
        # Get value of 'param' parameter.
        message = gce_protobuf_services.TrainingJobResultMessage()
        self.assertSetEqual(
            set(message.algorithm_id_to_message_mapping.keys()),
            set(vmconf.ALGORITHM_IDS))

    def test_that_classifier_data_is_stored_in_correct_attribute(self):
        """Ensures that classifier_data is stored correctly according to
        algorithm_id.
        """
        classifier_data = text_classifier_pb2.TextClassifier()
        classifier_data.model_json = 'dummy model'
        message = gce_protobuf_services.TrainingJobResultMessage()
        algorithm_id = 'TextClassifier'
        message.set_classifier_data(algorithm_id, classifier_data)
        self.assertEqual(
            message._message.WhichOneof('classifier_data'), # pylint: disable=protected-access
            message.algorithm_id_to_message_mapping[algorithm_id])

    def test_that_validation_checks_are_correct(self):
        """Ensure that validation checks work as expected."""
        message = gce_protobuf_services.TrainingJobResultMessage()
        with self.assertRaisesRegexp(
            Exception,
            'Expected classifier_data to be stored in text_classifier'):
            message.validate('TextClassifier')
