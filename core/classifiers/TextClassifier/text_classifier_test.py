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

"""Tests for text classifier."""

import json
import os

from core.classifiers.TextClassifier import TextClassifier
from core.tests import test_utils
import vmconf


def _load_training_data():
    file_path = os.path.join(vmconf.DATASETS_DIR,
                             'string_classifier_data.json')
    with open(file_path, 'r') as f:
        training_data = json.loads(f.read())
    return training_data


class TextClassifierTests(test_utils.GenericTestBase):
    """Tests for text classifier."""

    def setUp(self):
        super(TextClassifierTests, self).setUp()
        self.clf = TextClassifier.TextClassifier()
        self.training_data = _load_training_data()

    def test_that_text_classifier_works(self):
        """Test that entire classifier is working end-to-end."""
        self.clf.train(self.training_data)
        classifier_data = self.clf.to_dict()
        self.clf.validate(classifier_data)

    def test_text_classifier_performance(self):
        """Test the performance of the text classifier.

        This method measures and tests the run-time and the f1 score of the
        classifier. The run-time should be within 1 second and the f1 score
        should be greater than 0.85 for the test to pass.
        """
        self.clf.train(self.training_data)
        # The weighted f1 score for the test dataset should be at least 0.85.
        self.assertGreaterEqual(self.clf.best_score, 0.85)
        # The training phase for the test dataset should take less than 1 sec.
        self.assertLessEqual(self.clf.exec_time, 1)
