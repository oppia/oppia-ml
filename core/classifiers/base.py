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

"""Base class for classification algorithms"""

import abc


class BaseClassifier(object):
    """A base class for classifiers that uses supervised learning to match
    free-form text answers to answer groups. The classifier trains on answers
    that exploration editors have assigned to an answer group.

    Below are some concepts used in this class.
    training_data: list(dict). The training data that is used for training
        the classifier.
    label - An answer group that the training sample should correspond to. If a
        sample is being added to train a model, labels are provided.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @property
    @abc.abstractproperty
    def version(self):
        """Version of the classifier algorithm. The version of algorithm is
        matched with the version received as part of job data before training
        the classifier.
        """
        raise NotImplementedError

    @property
    @abc.abstractproperty
    def name_in_job_result_proto(self):
        """A property that identifies the attribute in job result proto message
        which will store this classifier's classifier data.
        """
        raise NotImplementedError

    @property
    @abc.abstractproperty
    def type_in_job_result_proto(self):
        """The type of the property in job result proto message which stores
        this classifier's classifier data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_proto(self):
        """Returns a protobuf of the frozen model consisting of trained
        parameters.

        Returns:
            Object. A protobuf object of frozen model containing trained
            model parameters. This data is used for prediction.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, training_data):
        """Trains classifier using given training_data.

        Args:
            training_data: list(dict). The training data that is used for
                training the classifier. The list contains dicts where each dict
                represents a single training data group, for example:
                training_data = [
                    {
                        'answer_group_index': 1,
                        'answers': [answer_1, answer_2]
                    },
                    {
                        'answer_group_index': 2,
                        'answers': [answer_3, answer_4]
                    }
                ]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self, frozen_model_proto):
        """Validates the specified frozen model protobuf object containing
        parameters of trained classifier model.

        Args:
            frozen_model_proto: Object of the frozen model protobuf containing
                parameters of trained classifier model.
        """
        raise NotImplementedError
