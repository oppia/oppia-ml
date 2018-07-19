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

import json
import os
import re

from core.classifiers import algorithm_registry
from core.classifiers import classifier_utils
from core.tests import test_utils
import vmconf

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
        expected_keys = [u'n_support', u'support_vectors', u'dual_coef',
                         u'intercept', u'classes', u'kernel_params', u'probA',
                         u'probB']
        self.assertListEqual(sorted(expected_keys), sorted(data.keys()))

        # Make sure that all of the values are of serializable type.
        self.assertEqual(type(data[u'n_support']), list)
        self.assertEqual(type(data[u'support_vectors']), list)
        self.assertEqual(type(data[u'dual_coef']), list)
        self.assertEqual(type(data[u'intercept']), list)
        self.assertEqual(type(data[u'classes']), list)
        self.assertEqual(type(data[u'probA']), list)
        self.assertEqual(type(data[u'probB']), list)
        self.assertEqual(type(data[u'kernel_params']), dict)
        self.assertEqual(type(data[u'kernel_params'][u'kernel']), unicode)
        self.assertEqual(type(data[u'kernel_params'][u'gamma']), float)
        self.assertEqual(type(data[u'kernel_params'][u'degree']), int)
        self.assertEqual(type(data[u'kernel_params'][u'coef0']), float)

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

    def test_that_float_verifier_regex_works_correctly(self):
        """Test that float verifier regex correctly identifies float values."""
        test_list = [
            '0.123', '+0.123', '-0.123', '.123', '+.123', '-.123',
            '0000003.1']
        for test in test_list:
            self.assertEqual(
                re.match(vmconf.FLOAT_VERIFIER_REGEX, test).groups()[0], test)

        test_list = [
            '3e10', '3e-10', '3e+10', '+3e10', '-3e10', '+3e+10', '+3e-10',
            '-3e-10', '-3e+10', '0.3e10', '-0.3e10', '+0.3e10', '-0.3e-10',
            '-0.3e+10', '+0.3e+10', '+0.3e-10', '.3e+10', '-.3e10']
        for test in test_list:
            self.assertEqual(
                re.match(vmconf.FLOAT_VERIFIER_REGEX, test).groups()[1], test)

        test_list = ['123', '0000', '123.']
        for test in test_list:
            self.assertIsNone(re.match(vmconf.FLOAT_VERIFIER_REGEX, test))

    def test_convert_float_numbers_to_string_in_classifier_data(self):
        """Make sure that all values are converted correctly."""
        test_dict = {
            'x': ['123', 'abc', 0.123]
        }

        expected_dict = {
            'x': ['123', 'abc', '0.123'],
        }

        output_dict = (
            classifier_utils.convert_float_numbers_to_string_in_classifier_data(
                test_dict))

        self.assertDictEqual(expected_dict, output_dict)

        test_dict = {
            'x': '-0.123'
        }

        with self.assertRaisesRegexp(
            Exception,
            'Float values should not be stored as strings.'):
            classifier_utils.convert_float_numbers_to_string_in_classifier_data(
                test_dict)


        test_dict = {
            'x': ['+0.123', 0.456]
        }

        with self.assertRaisesRegexp(
            Exception,
            'Float values should not be stored as strings.'):
            classifier_utils.convert_float_numbers_to_string_in_classifier_data(
                test_dict)

        test_dict = {
            'a': {
                'ab': 'abcd',
                'ad': {
                    'ada': 'abcdef',
                    'adc': [{
                        'adca': 'abcd',
                        'adcb': 0.1234,
                        'adcc': ['ade', 'afd']
                    }]
                },
                'ae': [['123', 0.123], ['abc']],
            },
            'b': {
                'bd': [-2.48521656693, -2.48521656693, -2.48521656693],
                'bg': ['abc', 'def', 'ghi'],
                'bh': ['abc', '123'],
            },
            'c': 1.123432,
        }

        expected_dict = {
            'a': {
                'ab': 'abcd',
                'ad': {
                    'ada': 'abcdef',
                    'adc': [{
                        'adca': 'abcd',
                        'adcb': '0.1234',
                        'adcc': ['ade', 'afd'],
                    }],
                },
                'ae': [['123', '0.123'], ['abc']],
            },
            'b': {
                'bd': ['-2.48521656693', '-2.48521656693', '-2.48521656693'],
                'bg': ['abc', 'def', 'ghi'],
                'bh': ['abc', '123'],
            },
            'c': '1.123432',
        }

        output_dict = (
            classifier_utils.convert_float_numbers_to_string_in_classifier_data(
                test_dict))
        self.assertDictEqual(expected_dict, output_dict)

    def test_that_pretrained_models_for_all_classifier_are_correct(self):
        """Make sure that trained classifier models generated in the output
        by each classifier do not raise Exception when passed through
        convert_float_numbers_to_string_in_classifier_data function."""

        classifier_ids = (
            algorithm_registry.Registry.get_all_classifier_algorithm_ids())
        for classifier_id in classifier_ids:
            file_path = os.path.join(
                vmconf.PRETRAINED_MODELS_PATH, '%s.json' % classifier_id)
            with open(file_path, 'r') as f:
                classifier_data = json.loads(f.read())
                output_dict = classifier_utils.convert_float_numbers_to_string_in_classifier_data( # pylint: disable=line-too-long
                    classifier_data)
                self.assertIsInstance(output_dict, dict)
