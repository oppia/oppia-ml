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

    @abc.abstractmethod
    def to_dict(self):
        """Returns a dict representing this classifier.

        Returns:
            dict. A dictionary representation of classifier referred as
            'classifier_data'. This data is used for prediction.
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
                        'answers': ['a1', 'a2']
                    },
                    {
                        'answer_group_index': 2,
                        'answers': ['a2', 'a3']
                    }
                ]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self, classifier_data):
        """Validates classifier data.

        Args:
            classifier_data: dict of the classifier attributes specific to
                the classifier algorithm used.
        """
        raise NotImplementedError
