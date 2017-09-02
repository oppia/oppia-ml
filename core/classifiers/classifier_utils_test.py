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

"""Tests for utility functions defined in classifier_utils."""

from core.classifiers import classifier_utils
from core.tests import test_utils

import numpy as np
from sklearn import svm


class ClassifierUtilsTest(test_utils.GenericTestBase):
    """Tests for utility functions."""

    def setUp(self):
        super(ClassifierUtilsTest, self).setUp()

        # Example training dataset.
        self.data = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        self.labels = np.array([1, 1, 0, 0])

    def test_that_svc_parameters_are_extracted_correctly(self):
        """Test that SVC classifier's parameters are extracted correctly."""
        clf = svm.SVC()
        # Train the model.
        clf.fit(self.data, self.labels)
        data = classifier_utils.extract_svm_parameters(clf)
        expected_keys = ['n_support', 'support_vectors', 'dual_coef',
            'n_support', 'intercept', 'classes',
            'kernel_params', 'probA', 'probB']
        self.assertListEqual(sorted(expected_keys), sorted(data.keys()))

        # Make sure that all of the values are of serializable type.
        self.assertEqual(type(data['n_support']), list)
        self.assertEqual(type(data['support_vectors']), list)
        self.assertEqual(type(data['dual_coef']), list)
        self.assertEqual(type(data['intercept']), list)
        self.assertEqual(type(data['classes']), list)
        self.assertEqual(type(data['probA']), list)
        self.assertEqual(type(data['probB']), list)
        self.assertEqual(type(data['kernel_params']), dict)
        self.assertEqual(type(data['kernel_params']['kernel']), unicode)
        self.assertEqual(type(data['kernel_params']['gamma']), float)
        self.assertEqual(type(data['kernel_params']['degree']), int)
        self.assertEqual(type(data['kernel_params']['coef0']), float)

    def check_that_unicode_validator_works_as_expected(self):
        """Make sure that unicode validator function works as expected."""
        test_dict = {
            'a': u'b',
            u'c': {
                u'abc': 20,
                u'cdf': [u'j', u'k']
            },
            u'x': [{u'm': u'n'}, {u'e': u'f'}]
        }

        with self.assertRaisesRegexp(
            Exception, 'Expected \'a\' to be unicode but found str.'):
            classifier_utils.unicode_validator_for_classifier_data(test_dict)

        test_dict = {
            u'a': 'b',
            u'c': {
                u'abc': 20,
                u'cdf': [u'j', u'k']
            },
            u'x': [{u'm': u'n'}, {u'e': u'f'}]
        }

        with self.assertRaisesRegexp(
            Exception, 'Expected \'b\' to be unicode but found str.'):
            classifier_utils.unicode_validator_for_classifier_data(test_dict)

        test_dict = {
            u'a': u'b',
            u'c': {
                'abc': 20,
                u'cdf': [u'j', u'k']
            },
            u'x': [{u'm': u'n'}, {u'e': u'f'}]
        }

        with self.assertRaisesRegexp(
            Exception, 'Expected \'abc\' to be unicode but found str.'):
            classifier_utils.unicode_validator_for_classifier_data(test_dict)

        test_dict = {
            u'a': u'b',
            u'c': {
                u'abc': 20,
                u'cdf': ['j', u'k']
            },
            u'x': [{u'm': u'n'}, {u'e': u'f'}]
        }

        with self.assertRaisesRegexp(
            Exception, 'Expected \'j\' to be unicode but found str.'):
            classifier_utils.unicode_validator_for_classifier_data(test_dict)

        test_dict = {
            u'a': u'b',
            u'c': {
                u'abc': 20,
                u'cdf': [u'j', u'k']
            },
            u'x': [{'m': u'n'}, {u'e': u'f'}]
        }

        with self.assertRaisesRegexp(
            Exception, 'Expected \'m\' to be unicode but found str.'):
            classifier_utils.unicode_validator_for_classifier_data(test_dict)

        test_dict = {
            u'a': u'b',
            u'c': {
                u'abc': 20,
                u'cdf': [u'j', u'k']
            },
            u'x': [{u'm': u'n'}, {u'e': 'f'}]
        }

        with self.assertRaisesRegexp(
            Exception, 'Expected \'f\' to be unicode but found str.'):
            classifier_utils.unicode_validator_for_classifier_data(test_dict)
