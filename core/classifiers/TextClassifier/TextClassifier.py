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

"""Classifier for free-form text answers."""

import json
import logging
import time

from sklearn import model_selection
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

from core.classifiers import base
from core.classifiers import classifier_utils


class TextClassifier(base.BaseClassifier):
    """A classifier that uses supervised learning to match free-form text
    answers to answer groups. The classifier trains on answers that exploration
    editors have assigned to an answer group. This classifier uses scikit's
    Support Vector Classifier (SVC) to obtain the best model using the linear
    kernel.
    """

    def __init__(self):
        super(TextClassifier, self).__init__()
        # sklearn.svm.SVC classifier object.
        self.best_clf = None

        # sklearn.feature_extraction.text.CountVectorizer object. It fits
        # text into a feature vector made up of word counts.
        self.count_vector = None

        # A dict representing the best parameters for the
        # sklearn.svm.SVC classifier.
        self.best_params = None

        # The f1 score of the best classifier found with GridSearch.
        self.best_score = None

        # Time taken to train the classifier
        self.exec_time = None

    @property
    def name_in_job_result_proto(self):
        return 'text_classifier'

    @property
    def type_in_job_result_proto(self):
        return '%sFrozenModel' % (self.__class__.__name__)

    def train(self, training_data):
        """Trains classifier using given training_data.

        Args:
            training_data: list(dict). The training data that is used for
                training the classifier. The list contains dicts where each
                dict represents a single training data group, for example:
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
        x = []
        y = []
        start = time.time()
        for answer_group in training_data:
            for answer in answer_group['answers']:
                x.append(answer)
                y.append(answer_group['answer_group_index'])

        count_vector = CountVectorizer()
        # Learn a vocabulary dictionary of all tokens in the raw documents.
        count_vector.fit(x)
        # Transform document to document-term matrix
        transformed_vector = count_vector.transform(x)

        # Set the range of parameters for the exhaustive grid search.
        param_grid = [{
            u'C': [0.5, 1, 10, 50, 100],
            u'kernel': [u'linear']
        }]

        clf = model_selection.GridSearchCV(
            svm.SVC(probability=True), param_grid, scoring='f1_weighted',
            n_jobs=-1)
        clf.fit(transformed_vector, y)
        end = time.time()
        logging.info(
            'The best score for GridSearch=%s', clf.best_score_)
        logging.info(
            'train() spent %f seconds for %d instances', end-start, len(x))
        self.best_params = clf.best_params_
        self.best_clf = clf.best_estimator_
        self.best_score = clf.best_score_
        self.count_vector = count_vector
        self.exec_time = end-start

    def to_dict(self):
        """Returns a dict representing this classifier.

        Returns:
            dict. A dictionary representation of classifier referred to
            as 'classifier_data'. This data is used for prediction.
        """
        classifier_data = {
            u'SVM': classifier_utils.extract_svm_parameters(self.best_clf),
            u'cv_vocabulary': self.count_vector.__dict__['vocabulary_'],
            u'best_params': self.best_params,
            u'best_score': self.best_score
        }
        return classifier_data

    # pylint: disable=too-many-branches
    def validate(self, classifier_data):
        """Validates classifier data.

        Args:
            classifier_data: dict of the classifier attributes specific to
                the classifier algorithm used.
        """
        allowed_top_level_keys = [u'SVM', u'cv_vocabulary', u'best_params',
                                  u'best_score']
        allowed_best_params_keys = [u'kernel', u'C']
        allowed_svm_kernel_params_keys = [u'kernel', u'gamma', u'coef0',
                                          u'degree']
        allowed_svm_keys = [u'n_support', u'dual_coef', u'support_vectors',
                            u'intercept', u'classes', u'kernel_params',
                            u'probA', u'probB']

        for key in allowed_top_level_keys:
            if key not in classifier_data:
                raise Exception(
                    '\'%s\' key not found in classifier_data.' % key)

            if key != u'best_score':
                if not isinstance(classifier_data[key], dict):
                    raise Exception(
                        'Expected  \'%s\' to be dict but found \'%s\'.'
                        % (key, type(classifier_data[key])))
            else:
                if not isinstance(classifier_data[key], float):
                    raise Exception(
                        'Expected  \'%s\' to be float but found \'%s\'.'
                        % (key, type(classifier_data[key])))

        for key in allowed_best_params_keys:
            if key not in classifier_data[u'best_params']:
                raise Exception(
                    '\'%s\' key not found in \'best_params\''
                    ' in classifier_data.' % key)

        for key in allowed_svm_keys:
            if key not in classifier_data[u'SVM']:
                raise Exception(
                    '\'%s\' key not found in \'SVM\''
                    ' in classifier_data.' % key)

        for key in allowed_svm_kernel_params_keys:
            if key not in classifier_data[u'SVM'][u'kernel_params']:
                raise Exception(
                    '\'%s\' key not found in \'kernel_params\''
                    ' in classifier_data.' % key)

        if not isinstance(classifier_data[u'best_params'][u'C'], float):
            raise Exception(
                'Expected \'C\' to be a float but found \'%s\'' %
                type(classifier_data[u'best_params'][u'C']))

        if not isinstance(classifier_data[u'best_params'][u'kernel'],
                          basestring):
            raise Exception(
                'Expected \'kernel\' to be a string but found \'%s\'' %
                type(classifier_data[u'best_params'][u'kernel']))

        # Validate that all the strings in classifier data are of unicode type.
        classifier_utils.unicode_validator_for_classifier_data(classifier_data)

        # Validate that entire classifier data is json serializable and
        # does not raise any exception.
        json.dumps(classifier_data)
