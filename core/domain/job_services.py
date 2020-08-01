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

"""This module contains functions used for polling, training and saving jobs."""

import logging

from core.classifiers import algorithm_registry
from core.domain import training_job_result_domain
from core.domain import remote_access_services

# pylint: disable=too-many-branches
def _validate_job_data(job_data):
    if not isinstance(job_data, dict):
        raise Exception('Invalid format of job data')

    if 'job_id' not in job_data:
        raise Exception('job data should contain job id')

    if 'training_data' not in job_data:
        raise Exception('job data should contain training data')

    if 'algorithm_id' not in job_data:
        raise Exception('job data should contain algorithm id')

    if 'algorithm_version' not in job_data:
        raise Exception('job data should contain algorithm version')

    if not isinstance(job_data['job_id'], unicode):
        raise Exception(
            'Expected job id to be unicode, received %s' %
            job_data['job_id'])

    if not isinstance(job_data['algorithm_id'], unicode):
        raise Exception(
            'Expected algorithm id to be unicode, received %s' %
            job_data['algorithm_id'])

    if not isinstance(job_data['algorithm_version'], int):
        raise Exception(
            'Expected algorithm version to be integer, received %s' %
            job_data['algorithm_version'])

    if not isinstance(job_data['training_data'], list):
        raise Exception(
            'Expected training data to be a list, received %s' %
            job_data['training_data'])

    algorithm_ids = (
        algorithm_registry.Registry.get_all_classifier_algorithm_ids())
    if job_data['algorithm_id'] not in algorithm_ids:
        raise Exception('Invalid algorithm id %s' % job_data['algorithm_id'])

    classifier = algorithm_registry.Registry.get_classifier_by_algorithm_id(
        job_data['algorithm_id'])
    if classifier.version != job_data['algorithm_version']:
        raise Exception(
            'Classifier version %d mismatches algorithm version %d received '
            'in job data' % (classifier.version, job_data['algorithm_version']))

    for grouped_answers in job_data['training_data']:
        if 'answer_group_index' not in grouped_answers:
            raise Exception(
                'Expected answer_group_index to be a key in training_data',
                ' list item')
        if 'answers' not in grouped_answers:
            raise Exception(
                'Expected answers to be a key in training_data list item')
        if not isinstance(grouped_answers['answer_group_index'], int):
            raise Exception(
                'Expected answer_group_index to be an int, received %s' %
                grouped_answers['answer_group_index'])
        if not isinstance(grouped_answers['answers'], list):
            raise Exception(
                'Expected answers to be a list, received %s' %
                grouped_answers['answers'])


def get_next_job():
    """Get next job request.

    Returns: dict. A dictionary containing job data.
    """
    job_data = remote_access_services.fetch_next_job_request()
    if job_data:
        _validate_job_data(job_data)
    return job_data


def train_classifier(algorithm_id, algorithm_version, training_data):
    """Train classifier associated with 'algorithm_id' using 'training_data'.

    Args:
        algorithm_id: str. ID of classifier algorithm.
        algorithm_version: int. Version of the classifier algorithm.
        training_data: list(dict). A list containing training data. Each dict
            stores 'answer_group_index' and 'answers'.

    Returns:
        FrozenModel. A protobuf object containing trained parameters of the
        classifier algorithm.
    """
    classifier = algorithm_registry.Registry.get_classifier_by_algorithm_id(
        algorithm_id)
    if classifier.version != algorithm_version:
        logging.warning(
            'Classifier version %d mismatches algorithm version %d received '
            'in job data', classifier.version, algorithm_version)
        return None
    classifier.train(training_data)
    frozen_model_proto = classifier.to_proto()
    classifier.validate(frozen_model_proto)
    return frozen_model_proto


def store_job_result(job_id, algorithm_id, frozen_model_proto):
    """Store result of job in the Oppia server.

    Args:
        job_id: str. ID of the job whose result is to be stored.
        algorithm_id: str. ID of the classifier algorithm.
        frozen_model_proto: Object. A protobuf object that stores trained model
            parameters.
    Returns:
        int. Status code of response.
    """
    job_result = training_job_result_domain.TrainingJobResult(
        job_id, algorithm_id, frozen_model_proto)
    status = remote_access_services.store_trained_classifier_model(
        job_result)
    return status
