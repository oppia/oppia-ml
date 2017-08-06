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

"""Tests for code classifier."""

import json
import os

from core.classifiers.CodeClassifier import CodeClassifier
from core.tests import test_utils
import vmconf


def _load_training_data():
    file_path = os.path.join(vmconf.DATASETS_DIR, 'CodeClassifier.json')
    with open(file_path, 'r') as f:
        training_data = json.loads(f.read())
    return training_data


class CodeClassifierTests(test_utils.GenericTestBase):
    """Tests for code classifier and related preprocessing functions."""

    def setUp(self):
        super(CodeClassifierTests, self).setUp()
        # Example programs for testing preprocessing functions.
        self.program_a = (
            '# In Python, the code\n#\n#     for letter in [\'a\', \'b\']:\n#'
            '         print letter\n#\n# prints:\n#\n#     a\n#     '
            'b\nsum = 0\nfor num in range(1000):\n  if num%3==0 and num%5==0:'
            '\n\tsum = sum + num\n  else:\n\tif num%3 == 0:\n\t  '
            'sum = sum + num\n\tif num%5 == 0:\n\t  sum = sum + num\nprint sum')

        self.program_b = (
            '# In Python, the code\n#\n#     for letter in [\'a\', \'b\']:\n#'
            '         print letter\n#\n# prints:\n#\n#     a\n#     b\n\n'
            'for num in range(1000):\n  if num % 3 == 0 or num % 5 == 0:\n'
            '    sum += num\nprint sum')

        self.data = {
            1: {
                'source': self.program_a,
                'class': 1
            },
            2: {
                'source': self.program_b,
                'class': 2
            }
        }

        self.clf = CodeClassifier.CodeClassifier()

    def test_that_correct_tokens_are_generated(self):
        """Make sure that get_tokens function returns correct tokens."""
        tokens = list(CodeClassifier.get_tokens(self.program_a))
        expected_tokens = [
            (53, '# In Python, the code'), (54, '\n'), (53, '#'), (54, '\n'),
            (53, "#     for letter in ['a', 'b']:"), (54, '\n'),
            (53, '#         print letter'), (54, '\n'), (53, '#'), (54, '\n'),
            (53, '# prints:'), (54, '\n'), (53, '#'), (54, '\n'),
            (53, '#     a'), (54, '\n'), (53, '#     b'), (54, '\n'),
            (1, 'sum'), (51, '='), (2, '0'), (4, '\n'), (1, 'for'), (1, 'num'),
            (1, 'in'), (1, 'range'), (51, '('), (2, '1000'), (51, ')'),
            (51, ':'), (4, '\n'), (5, '  '), (1, 'if'), (1, 'num'), (51, '%'),
            (2, '3'), (51, '=='), (2, '0'), (1, 'and'), (1, 'num'), (51, '%'),
            (2, '5'), (51, '=='), (2, '0'), (51, ':'), (4, '\n'), (5, '\t'),
            (1, 'sum'), (51, '='), (1, 'sum'), (51, '+'), (1, 'num'), (4, '\n'),
            (6, ''), (1, 'else'), (51, ':'), (4, '\n'), (5, '\t'), (1, 'if'),
            (1, 'num'), (51, '%'), (2, '3'), (51, '=='), (2, '0'), (51, ':'),
            (4, '\n'), (5, '\t  '), (1, 'sum'), (51, '='), (1, 'sum'),
            (51, '+'), (1, 'num'), (4, '\n'), (6, ''), (1, 'if'), (1, 'num'),
            (51, '%'), (2, '5'), (51, '=='), (2, '0'), (51, ':'), (4, '\n'),
            (5, '\t  '), (1, 'sum'), (51, '='), (1, 'sum'), (51, '+'),
            (1, 'num'), (4, '\n'), (6, ''), (6, ''), (6, ''), (1, 'print'),
            (1, 'sum'), (0, '')]

        self.assertListEqual(expected_tokens, tokens)

    def test_that_tokenize_for_cv_works(self):
        """Make sure that custom tokenizer used for CountVectorizer is
        working as expected."""
        tokens = CodeClassifier.tokenize_for_cv(self.program_a)
        expected_tokens = [
            'V', '=', '0', 'for', 'V', 'in', 'V', '(', '1000', ')', ':', 'if',
            'V', '%', '3', '==', '0', 'and', 'V', '%', '5', '==', '0', ':', 'V',
            '=', 'V', '+', 'V', 'else', ':', 'if', 'V', '%', '3', '==', '0',
            ':', 'V', '=', 'V', '+', 'V', 'if', 'V', '%', '5', '==', '0', ':',
            'V', '=', 'V', '+', 'V', 'print', 'V']

        self.assertListEqual(tokens, expected_tokens)

    def test_that_token_to_id_is_correct(self):
        """Make sure that correct token_to_id map is generated."""
        token_to_id = CodeClassifier.map_tokens_to_ids(self.data, 0)
        expected_tokens = [
            'and', 'UNK', '%', 'for', ')', '(', '+', 'V', 'else', '==', '0',
            '3', '5', '1000', 'in', 'print', ':', '=', 'or', '+=', 'if']
        self.assertListEqual(token_to_id.keys(), expected_tokens)

    def test_that_jaccard_index_is_calculated_correctly(self):
        """Make sure that correct jaccard index is calculated between
        two sets."""
        # Check for single element sets.
        set_a = [1]
        set_b = [2]
        expected_index = 0.0
        jaccard_index = CodeClassifier.calc_jaccard_index(set_a, set_b)
        self.assertEqual(expected_index, jaccard_index)

        set_a = [1]
        set_b = [1]
        expected_index = 1.0 / 1.0
        jaccard_index = CodeClassifier.calc_jaccard_index(set_a, set_b)
        self.assertEqual(expected_index, jaccard_index)

        # Check for normal sets.
        set_a = [1]
        set_b = [1, 2]
        expected_index = 1.0 / 2.0
        jaccard_index = CodeClassifier.calc_jaccard_index(set_a, set_b)
        self.assertEqual(expected_index, jaccard_index)

        set_a = [2, 3, 4]
        set_b = [1, 2, 4, 6]
        expected_index = 2.0 / 5.0
        jaccard_index = CodeClassifier.calc_jaccard_index(set_a, set_b)
        self.assertEqual(expected_index, jaccard_index)

        # Check for multisets.
        set_a = [1, 2, 2, 3]
        set_b = [2, 3, 4]
        expected_index = 2.0 / 5.0
        jaccard_index = CodeClassifier.calc_jaccard_index(set_a, set_b)
        self.assertEqual(expected_index, jaccard_index)

    def test_that_code_classifier_works(self):
        """Test that entire classifier is working from end-to-end."""
        training_data = _load_training_data()
        self.clf.train(training_data)
        classifier_data = self.clf.to_dict()
        self.clf.validate(classifier_data)
