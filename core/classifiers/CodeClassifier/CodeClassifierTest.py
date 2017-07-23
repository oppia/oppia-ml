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

from core.classifiers.CodeClassifier import CodeClassifier
from core.tests import test_utils

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

    def test_that_cv_tokenizer_works(self):
        """Make sure that custom tokenizer used for CountVectorizer is
        working as expected."""
        tokens = CodeClassifier.cv_tokenizer(self.program_a)
        expected_tokens = [
            'V', '=', '0', 'for', 'V', 'in', 'V', '(', '1000', ')', ':', 'if',
            'V', '%', '3', '==', '0', 'and', 'V', '%', '5', '==', '0', ':', 'V',
            '=', 'V', '+', 'V', 'else', ':', 'if', 'V', '%', '3', '==', '0',
            ':', 'V', '=', 'V', '+', 'V', 'if', 'V', '%', '5', '==', '0', ':',
            'V', '=', 'V', '+', 'V', 'print', 'V']

        self.assertListEqual(tokens, expected_tokens)

    def test_that_token_to_id_is_correct(self):
        """Make sure that correct token_to_id map is generated."""
        token_to_id = CodeClassifier.generate_token_to_id(self.data, 0)
        expected_tokens = [
            'and', 'UNK', '%', 'for', ')', '(', '+', 'V', 'else', '==', '0',
            '3', '5', '1000', 'in', 'print', ':', '=', 'or', '+=', 'if']
        self.assertListEqual(token_to_id.keys(), expected_tokens)

    def test_that_hash_generator_works(self):
        """Make sure that hash generator function works as expected."""
        token_to_id = CodeClassifier.generate_token_to_id(self.data, 0)
        id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
        tokens = [id_to_token[0], id_to_token[1], id_to_token[2]]
        hash_value = CodeClassifier.hash_generator(token_to_id, tokens)

        n = len(token_to_id)
        expected_hash_value = 0 * (n ** 2) + 1 * (n ** 1) + 2 * (n ** 0)

        self.assertEqual(hash_value, expected_hash_value)

    def test_that_k_gram_hash_generator_works(self):
        """Make sure that k-gram hash generator function works as expected."""
        token_to_id = CodeClassifier.generate_token_to_id(self.data, 0)
        id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
        tokens = [
            id_to_token[0], id_to_token[1], id_to_token[2], id_to_token[3],
            id_to_token[4], id_to_token[5]]
        k_grams = CodeClassifier.k_gram_hash_generator(tokens, token_to_id, 3)
        expected_k_grams = [23, 486, 949, 1412]
        self.assertListEqual(expected_k_grams, k_grams)

    def test_that_correct_fingerprints_are_obtained(self):
        """Make sire that fingerprint generator generates correct fingerprint.
        """
        token_to_id = CodeClassifier.generate_token_to_id(self.data, 0)
        id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
        tokens = [
            id_to_token[0], id_to_token[1], id_to_token[2], id_to_token[3],
            id_to_token[4], id_to_token[5]]
        k_grams = CodeClassifier.k_gram_hash_generator(tokens, token_to_id, 3)
        fingerprint = CodeClassifier.get_fingerprint_from_hashes(k_grams, 3)
        expected_fingerprint = [(486, 1), (23, 0)]
        self.assertListEqual(fingerprint, expected_fingerprint)

    def test_that_jaccard_index_is_calculated_correctly(self):
        """Make sure that correct jaccard index is calculated between
        two sets."""
        set1 = [1, 2, 2, 3]
        set2 = [2, 3, 4]
        expected_index = 0.4
        jaccard_index = CodeClassifier.calc_jaccard_index(set1, set2)
        self.assertEqual(expected_index, jaccard_index)
