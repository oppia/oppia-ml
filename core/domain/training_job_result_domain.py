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

from core.classifiers import algorithm_registry

class TrainingJobResult(object):
    """TrainingJobResult domain object."""

    def __init__(self, job_id, algorithm_id, classifier_data):
        """Initializes TrainingJobResult object.

        Args:
            job_id: str. The id of the training job whose results are stored
                in classifier_data.
            algorithm_id: str. The id of the algorithm of the training job.
            classifier_data: object. Frozen model of the corresponding
                training job.
        """
        self.job_id = job_id
        self.algorithm_id = algorithm_id
        self.classifier_data = classifier_data

    def validate(self):
        """Validate that TrainigJobResult object stores correct data.

        Raises:
            Exception: str. If the classifier data is stored in a field
                that does not correspond to algorithm_id.
        """
        classifier = algorithm_registry.Registry.get_classifier_by_algorithm_id(
            self.algorithm_id)
        if (
                type(self.classifier_data).__name__ !=
                classifier.type_in_job_result_proto):
            raise Exception(
                "Expected classifier data of type %s but found %s type" % (
                    classifier.type_in_job_result_proto,
                    type(self.classifier_data).__name__))
