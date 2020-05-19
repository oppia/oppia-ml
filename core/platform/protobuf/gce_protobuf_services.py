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

"Functions and class definitions related to protobuf files used in Oppia-ml"

from core.platform.protobuf.protofiles import training_job_data_pb2 # pylint: disable=no-name-in-module

class TrainingJobResultMessage(object):
    """TrainingJobDataMessage used to transfer trained classifier and relevant
    job data to Oppia."""

    def __init__(self):
        self._message = training_job_data_pb2.TrainingJobData.JobResult()
        self.algorithm_id_to_message_mapping = {
            'TextClassifier': 'text_classifier'
        }

    def set_job_id(self, job_id):
        """Sets job_id of the training job.

        Args:
            job_id: str. Job id of the training job.
        """
        self._message.job_id = job_id

    def set_classifier_data(self, algorithm_id, classifier_data):
        """Adds classifier data to training job data message according to
        algorithm_id.

        Args:
            algorithm_id: str. The id of the algorithm of the training job.
            classifier_data: object. The classifier data that is exported by
                algorithm.
        """
        classifier_data_attribute = self.algorithm_id_to_message_mapping[
            algorithm_id]
        getattr(self._message, classifier_data_attribute).CopyFrom(
            classifier_data)

    def validate(self, algorithm_id):
        """Validate that classifier data is correctly stored.

        Args:
            algorithm_id: str. The id of the algorithm of the training job.

        Raises:
            Exception: str. If the classifier data is stored in a field
                that does not correspond to algorithm_id.
        """
        classifier_data_attribute = self.algorithm_id_to_message_mapping[
            algorithm_id]

        if self._message.WhichOneof('classifier_data') != (
                classifier_data_attribute):
            raise Exception(
                "Expected classifier_data to be stored in "
                "%s but is stored in %s" % (
                    classifier_data_attribute,
                    self._message.WhichOneof('classifier_data')))
