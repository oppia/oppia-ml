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

"""Functions for training job result domain object and protobuf message."""

from core.classifiers import algorithm_registry
from core.domain.proto import training_job_response_payload_pb2

def get_proto_message_from_training_job_result(job_result):
    """Generate TrainingJobResult protobuf object from the TrainingJobResult
    domain object.

    Args:
        job_result: TrainingJobResult. TrainingJobResult domain object
            containing training job result data.

    Returns:
        TrainingJobResult protobuf object. Protobuf object corresponding to
            TrainingJobResult protobuf message definition.
    """
    job_result.validate()
    proto_message = (
        training_job_response_payload_pb2.TrainingJobResponsePayload.JobResult()) # pylint: disable=line-too-long
    proto_message.job_id = job_result.job_id
    job_result_attribute = (
        algorithm_registry.Registry.get_classifier_by_algorithm_id(
            job_result.algorithm_id).name_in_job_result_proto)
    getattr(proto_message, job_result_attribute).CopyFrom(
        job_result.classifier_data)
    return proto_message
