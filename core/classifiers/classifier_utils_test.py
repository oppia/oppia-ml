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
        expected_keys = [
            'n_support', 'support_vectors', 'dual_coef', 'intercept', 'classes']
        self.assertListEqual(sorted(expected_keys), sorted(data.keys()))

        # Make sure that all of the values are of list type.
        for key in data:
            self.assertEqual(type(data[key]), list)
