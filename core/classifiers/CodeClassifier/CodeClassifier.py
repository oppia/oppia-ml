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

from collections import Counter
import keyword
from StringIO import StringIO
import token
import tokenize

from core.classifiers.BaseClassifier import BaseClassifier

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# pylint: disable=invalid-name
def _get_tokens(program):
    """Generate tokens for program using tokenize module."""
    for tid, tname, _, _, _ in tokenize.generate_tokens(
            StringIO(program).readline):
        yield (tid, tname)


def _cv_tokenizer(program):
    """Tokenize Python program for CountVectorizer."""
    token_program = []
    for tid, tname in _get_tokens(program):
        if tid == token.N_TOKENS or tid == 54:
            continue
        elif tid == token.NAME:
            if tname in keyword.kwlist:
                token_program.append(tname)
            else:
                token_program.append('V')
        else:
            token_program.append(tname)

    return token_program


def _generate_token_to_id(data, threshold=5):
    """Generates a list of valid tokens and assigns a unique ID to each
    token."""
    # All unique tokens and number of time they occur in dataset. A token
    # can be a keyword, an identifier, a constant, an operator.
    alphabet = {}

    for pid in data:
        program = data[pid]['source']
        for tid, tname in _get_tokens(program):
            # If tid is tokens.NAME then only add if it is a python keyword.
            # Treat all variables and methods same.
            if tid == token.N_TOKENS or tid == 54:
                continue
            elif tid == token.NAME:
                if tname in keyword.kwlist:
                    alphabet[tname] = alphabet.get(tname, 0) + 1
            else:
                alphabet[tname] = alphabet.get(tname, 0) + 1

    # Consider only those tokens which occur for more than threshold times
    # in entire dataset.
    valid_tokens = [k for k, v in alphabet.iteritems() if v > threshold]
    token_to_id = dict(zip(valid_tokens, xrange(0, len(valid_tokens))))

    # Add 'UNK' in token_to_id. This will be used to replace any token
    # occurring in program which is not in valid_token.
    token_to_id['UNK'] = len(token_to_id)

    # Add 'V' in token_to_id. This token will be used to replace all
    # variables and methods in program.
    token_to_id['V'] = len(token_to_id)
    return token_to_id


def _tokenize_data(data):
    """Tokenize Python programs in dataset for winnowing."""
    token_to_id = _generate_token_to_id(data)

    # Tokenize all programs in dataset.
    for program_id in data:
        program = data[program_id]['source']
        token_program = []
        for tid, tname in _get_tokens(program):
            if tid == token.N_TOKENS or tid == 54:
                continue
            elif tid == token.NAME:
                if tname in keyword.kwlist:
                    if tname not in token_to_id:
                        token_program.append('UNK')
                    else:
                        token_program.append(tname)
                else:
                    token_program.append('V')
            else:
                if tname not in token_to_id:
                    token_program.append('UNK')
                else:
                    token_program.append(tname)

        data[program_id]['tokens'] = token_program

    return data, token_to_id


def _hash_generator(token_to_id, tokens):
    """Generate hash for tokens in 'tokens' using token_to_id."""
    hash_val = 0
    n = len(tokens) - 1
    for x in tokens:
        hash_val += token_to_id[x] * (len(token_to_id) ** n)
        n -= 1
    return hash_val


def _k_gram_hash_generator(token_program, token_to_id, K):
    """Generate all k-gram hashes for tokenized program."""
    generated_hashes = [
        _hash_generator(token_to_id, token_program[i: i+K])
        for i in xrange(0, len(token_program) - K)]
    return generated_hashes


def _generate_k_gram_hashes(data, token_to_id, K=3):
    """Generate k-gram hashes for all programs in dataset."""
    for program_id in data:
        data[program_id]['k_gram_hashes'] = _k_gram_hash_generator(
            data[program_id]['tokens'], token_to_id, K)
    return data


def _get_fingerprint_from_hashes(k_gram_hashes, window_size):
    """Generate document fingerprint from k-gram hashes of given program."""
    generated_fingerprint = set()
    for i in xrange(0, len(k_gram_hashes) - window_size):
        window_hashes = k_gram_hashes[i: i + window_size]
        min_hash_index = i + min(
            xrange(window_size), key=window_hashes.__getitem__)
        min_hash = k_gram_hashes[min_hash_index]
        generated_fingerprint.add((min_hash, min_hash_index))

    return list(generated_fingerprint)


def _generate_program_fingerprints(data, T, K):
    """Generate document fingerprints for all programs in entire dataset."""
    window_size = T - K + 1
    for program_id in data:
        data[program_id]['fingerprint'] = _get_fingerprint_from_hashes(
            data[program_id]['k_gram_hashes'], window_size)
    return data


def _calc_jaccard_index(A, B):
    """Calculate jaccard's coefficient for two sets A and B."""
    small_set = A[:] if len(A) < len(B) else B[:]
    union_set = B[:] if len(A) < len(B) else A[:]
    for elem in union_set:
        if elem in small_set:
            small_set.remove(elem)
    union_set.extend(small_set)

    if union_set == []:
        return 0

    small_set = A[:] if len(A) < len(B) else B[:]
    intersection_set = []
    for elem in small_set:
        if elem in A and elem in B:
            intersection_set.append(elem)
            A.remove(elem)
            B.remove(elem)

    coeff = float(len(intersection_set)) / len(union_set)
    return coeff


def _get_program_similarity(fingerprint_a, fingerprint_b):
    """Find similarity between fingerprint of two programs."""
    A = [h for (h, _) in fingerprint_a]
    B = [h for (h, _) in fingerprint_b]
    return _calc_jaccard_index(A, B)


def _generate_top_similars(data, top):
    """Find 'top' nearest neighbours for all programs in dataset."""
    for program_id_1 in data:
        overlaps = []
        for program_id_2 in data:
            overlap = _get_program_similarity(
                data[program_id_1]['fingerprint'],
                data[program_id_2]['fingerprint'])
            overlaps.append((program_id_2, overlap))
        data[program_id_1]['top_similar'] = sorted(
            overlaps, key=lambda e: e[1], reverse=True)[:top]
    return data


def _run_knn(data):
    """Predict classes for each program in dataset using KNN."""

    # No. of times a class has to appear in nearest neighbours of a program
    # so that prediction is correct.
    occurrence = 0

    # Keep record of missclassified programs' ID.
    missclassified_points = []

    for pid in data:
        similars = data[pid]['top_similar']
        if pid in similars:
            similars.remove(pid)
        nearest_classes = [data[i]['class'] for (i, _) in similars]
        cnt = Counter(nearest_classes)
        common = cnt.most_common(1)
        data[pid]['prediction'] = common[0][0]

        if data[pid]['prediction'] == data[pid]['class']:
            # If prediction is correct then add no. of times the correct
            # class has to appear in nearest neighbours to occurrence.
            occurrence += common[0][1]
        else:
            # Else mark the ID fo program so that it will be
            # trained using SVM later.
            missclassified_points.append(pid)

    # Calculate average of occurrence. This is used during prediction so
    # that if winner of nearest neighbours appears less than the occurrence
    # we consider that KNN has failed in prediction.
    occurrence /= float(len(data))

    return occurrence, missclassified_points


# pylint: disable=too-many-instance-attributes, attribute-defined-outside-init
class CodeClassifier(BaseClassifier.BaseClassifierClass):
    """A class for code classifier that uses supervised learning to match
    Python programs to an answer group. The classifier trains on programs
    that exploration editors have assigned to an answer group.
    """
    # pylint: disable=useless-super-delegation
    def __init__(self):
        super(CodeClassifier, self).__init__()

    # pylint: enable=useless-super-delegation

    def to_dict(self):
        """Returns a dict representing this classifier.

        Returns:
            dict. A dictionary representation of classifier referred as
            'classifier_data'. This data is used for prediction.
        """
        winnowing_data = {
            i: {
                'fingerprint': self.data[i]['fingerprint'],
                'class': self.data[i]['class']
            } for i in self.data
        }

        classifier_data = {
            'KNN': {
                'token_to_id': self.token_to_id,
                'T': self.T,
                'K': self.K,
                'top': self.top,
                'occurrence': self.occurrence,
                'data': winnowing_data
            },
            'SVM': self.clf.__dict__,
            'class_to_answer_group_mapping': self.class_to_answer_group_mapping
        }
        return classifier_data

    # pylint: disable=too-many-locals
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
        # Answer group index to classifier class mapping.
        answer_group_to_class_mapping = {
            training_data[i]['answer_group_index']: i
            for i in range(len(training_data))
        }

        class_to_answer_group_mapping = dict(zip(
            answer_group_to_class_mapping.values(),
            answer_group_to_class_mapping.keys()))

        data = {}
        count = 0
        for answer_group in training_data:
            answer_group_index = answer_group['answer_group_index']
            for answer in answer_group['answers']:
                data[count] = {
                    'source': answer,
                    'class': answer_group_to_class_mapping[answer_group_index]
                }
                count += 1

        data, token_to_id = _tokenize_data(data)

        # No. of nearest neighbours to consider for classification using KNN.
        top = int(len(answer_group_to_class_mapping) / 2)

        # T is minimum length of a substring that winnowing should match when
        # comparing two programs.
        # K is length of substring for which hash value is to be generated. Each
        # substring of length K in program will be converted to equivalent hash
        # value.
        # Maximum allowed value of T.
        T_max = 10

        # GRID search for best value of T and K.
        results = []
        for T in range(4, T_max):
            for K in range(3, T):
                data = _generate_k_gram_hashes(data, token_to_id, K)
                data = _generate_program_fingerprints(data, T, K)
                data = _generate_top_similars(data, top)
                occurrence, missclassified_points = _run_knn(data)
                accuracy = (
                    len(data) - len(missclassified_points)) / float(len(data))
                results.append((accuracy, T, K))

        best_score = sorted(results, key=lambda e: e[0], reverse=True)[0]
        T = best_score[1]
        K = best_score[2]

        # Run KNN for best value of T and K.
        data = _generate_k_gram_hashes(data, token_to_id, K)
        data = _generate_program_fingerprints(data, T, K)
        data = _generate_top_similars(data, top)
        occurrence, missclassified_points = _run_knn(data)

        # Create a new dataset for missclassified points.
        programs = [data[pid]['source'] for pid in sorted(data.keys())]

        # Build vocabulary for programs in dataset. This vocabulary will be
        # used to generate Bag-of-Words vector for python programs.
        cv = CountVectorizer(tokenizer=_cv_tokenizer, min_df=5)
        cv.fit(programs)

        # Get BoW vectors for missclassified python programs.
        program_vecs = cv.transform(programs)
        # Store BoW vector for each program in data.
        for (i, pid) in enumerate(sorted(data.keys())):
            data[pid]['vector'] = program_vecs[i].todense()

        train_data = np.array(
            [data[pid]['vector'] for pid in data])
        train_result = np.array(
            [data[pid]['class'] for pid in data])

        # Generate sample weight. It assigns relative weight to each sample.
        # Higher weights force the classifier to put more emphasis on these
        # points.
        sample_weight = np.ones(len(data))
        # Increase weight of missclassified points.
        sample_weight[missclassified_points] = 3

        # Fix dimension of train_data. Sometime there is an extra redundant
        # axis generated when list is transformed in array.
        train_data = np.squeeze(train_data)

        # Search for best SVM estimator using grid search.
        param_grid = [{
            'C': [1, 3, 5, 8, 12],
            'kernel': ['rbf'],
            'gamma': [0.001, 0.01, 3, 7]
        }]

        fit_params = {
            'sample_weight': sample_weight
        }
        search = GridSearchCV(SVC(), param_grid, fit_params=fit_params)
        search.fit(train_data, train_result)
        clf = search.best_estimator_


        # Set attributes and their values.
        self.data = data
        self.token_to_id = token_to_id
        self.T = T
        self.K = K
        self.top = top
        self.occurrence = occurrence
        self.clf = clf
        self.class_to_answer_group_mapping = class_to_answer_group_mapping

    # pylint: enable=too-many-locals
    # pylint: disable=too-many-branches, no-self-use
    def validate(self, classifier_data):
        """Validates classifier data.

        Args:
            classifier_data: dict of the classifier attributes specific to
                the classifier algorithm used.
        """
        allowed_top_level_keys = ['KNN', 'SVM', 'class_to_answer_group_mapping']
        for key in allowed_top_level_keys:
            if key not in classifier_data:
                raise Exception(
                    '\'%s\' key not found in classifier_data.' % key)

        allowed_knn_keys = ['T', 'K', 'top', 'occurrence',
                            'token_to_id', 'data']
        for key in allowed_knn_keys:
            if key not in classifier_data['KNN']:
                raise Exception(
                    '\'%s\' key not found in \'KNN\' in classifier_data.' % key)

        if not isinstance(classifier_data['KNN']['T'], int):
            raise Exception(
                'Expected \'T\' to be an int but found \'%s\'' %
                type(classifier_data['KNN']['T']))

        if not isinstance(classifier_data['KNN']['K'], int):
            raise Exception(
                'Expected \'K\' to be an int but found \'%s\'' %
                type(classifier_data['KNN']['K']))

        if not isinstance(classifier_data['KNN']['top'], int):
            raise Exception(
                'Expected \'top\' to be an int but found \'%s\'' %
                type(classifier_data['KNN']['top']))

        if not isinstance(classifier_data['KNN']['occurrence'], float):
            raise Exception(
                'Expected \'occurrence\' to be a float but found \'%s\'' %
                type(classifier_data['KNN']['occurrence']))

        if not isinstance(classifier_data['KNN']['data'], dict):
            raise Exception(
                'Expected \'data\' to be a dict but found \'%s\'' %
                type(classifier_data['KNN']['data']))

        if not isinstance(classifier_data['KNN']['token_to_id'], dict):
            raise Exception(
                'Expected \'token_to_id\' to be a dict but found \'%s\'' %
                type(classifier_data['KNN']['token_to_id']))

        for pid in classifier_data['KNN']['data']:
            if 'fingerprint' not in classifier_data['KNN']['data'][pid]:
                raise Exception(
                    'No fingerprint found for program with \'%s\' pid in data.'
                    % pid)

            if 'class' not in classifier_data['KNN']['data'][pid]:
                raise Exception(
                    'No class found for program with \'%s\' pid in data.' % pid)
