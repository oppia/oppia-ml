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

"""Base class for classification algorithms."""

import collections
import keyword
import StringIO
import token
import tokenize

from core.classifiers import base

import numpy as np
from sklearn.feature_extraction import text as sklearn_text
from sklearn import model_selection
from sklearn import svm

# Token which will be used to replace variable and method name when tokenizing
# program.
VAR_TOKEN = 'V'

# Token which will be used to replace unkownn token when tokenizing
# program.
UNK_TOKEN = 'UNK'

# T is minimum length of a substring that winnowing should match when
# comparing two programs.
# Minimum value of T. T is model parameter of winnowing algorithm.
T_MIN = 3
# Maximum value of T. T is model parameter of winnowing algorithm.
T_MAX = 11

# K is length of substring for which hash value is to be generated. Each
# substring of length K in program will be converted to equivalent hash
# value (K <= T).
# Minimum value of K (K <= T). K is model parameter of winnowing algorithm.
K_MIN = 3

def get_token(program):
    """Generate tokens for program using tokenize module.

    Args:
        program: str. Input program for which tokens are to be generated.

    Yields:
        int. Integer value which represents type of token.
        str. Token.
    """
    for token_id, token_name, _, _, _ in tokenize.generate_tokens(
            StringIO.StringIO(program).readline):
        yield (token_id, token_name)


def cv_tokenizer(program, var_token=VAR_TOKEN):
    """Custom tokenizer of Python program for CountVectorizer in sci-kit.

    Args:
        program: str. Input program for which tokens are to be generated.
        var_token: str. Replace variable and method names with this string when
            tokenizing data.

    Returns:
        list(str). A list containing tokens of input program.
    """
    token_program = []
    for token_id, token_name in get_token(program):
        # Ignore all newline, comment and empty string tokens.
        if (token_id == token.N_TOKENS or token_id == 54
                or token_name.strip() == ''):
            continue
        elif token_id == token.NAME:
            # If token_id is tokens.NAME then only add if it is a python
            # keyword. Treat all variables and methods same.
            if token_name in keyword.kwlist:
                token_program.append(token_name)
            else:
                token_program.append(var_token)
        else:
            token_program.append(token_name)

    return token_program


def generate_token_to_id(
        data, threshold, var_token=VAR_TOKEN, unk_token=UNK_TOKEN):
    """Generates a list of valid tokens and assigns a unique ID to each
    token.

    Args:
        data: dict. A dictionary containing training_data.
        threshold: int. Ignore a token if it appears less than 'threshold' times
            in dataset.
        var_token: str. Replace variable and method names with this string when
            tokenizing data.
        unk_token: ste. Replace unknown tokens (ignored tokens) with this string
            when tokenizing the data.

    Returns:
        dict. A dictionary containing tokens mapped to a unique ID.
    """
    # All unique tokens and number of time they occur in dataset. A token
    # can be a keyword, an identifier, a constant, an operator.
    alphabet = {}

    for pid in data:
        program = data[pid]['source']
        for token_id, token_name in get_token(program):
            # Ignore all newline, comment and empty string tokens.
            if (token_id == token.N_TOKENS or token_id == 54
                    or token_name.strip() == ''):
                continue
            # If token_id is tokens.NAME then only add if it is a python
            # keyword.
            elif token_id == token.NAME:
                if token_name in keyword.kwlist:
                    alphabet[token_name] = alphabet.get(token_name, 0) + 1
            else:
                alphabet[token_name] = alphabet.get(token_name, 0) + 1

    # Consider only those tokens which occur for more than threshold times
    # in entire dataset.
    valid_tokens = [k for k, v in alphabet.iteritems() if v > threshold]
    token_to_id = dict(zip(valid_tokens, xrange(0, len(valid_tokens))))

    # Add 'UNK' in token_to_id. This will be used to replace any token
    # occurring in program which is not in valid_token.
    token_to_id[unk_token] = len(token_to_id)

    # Add 'V' in token_to_id. This token will be used to replace all
    # variables and methods in program.
    token_to_id[var_token] = len(token_to_id)
    return token_to_id


def tokenize_data(data, threshold=5, var_token=VAR_TOKEN, unk_token=UNK_TOKEN):
    """Tokenize Python programs in dataset for winnowing.

    Args:
        data: dict. A dictionary containing training_data.
        threshold: int. Ignore a token if it appears less than 'threshold' times
            in dataset.
        var_token: str. Replace variable and method names with this string when
            tokenizing data.
        unk_token: ste. Replace unknown tokens (ignored tokens) with this string
            when tokenizing the data.

    Returns:
        dict. A dictionary containing training data with additional key 'tokens'
        for each data point which stores tokens of corresponding data point.
    """
    token_to_id = generate_token_to_id(data, threshold)

    # Tokenize all programs in dataset.
    for program_id in data:
        program = data[program_id]['source']
        token_program = []
        for token_id, token_name in get_token(program):
            # Ignore all newline, comment and empty string tokens.
            if (token_id == token.N_TOKENS or token_id == 54
                    or token_name.strip() == ''):
                continue
            elif token_id == token.NAME:
                if token_name in keyword.kwlist:
                    # If token_id is tokens.NAME then only add if it is a python
                    # keyword and is present in token_to_id else replace it
                    # with UNK_TOKEN.
                    if token_name in token_to_id:
                        token_program.append(token_name)
                    else:
                        token_program.append(unk_token)
                else:
                    # If token_id is tokens.NAME and it is not a python keyword
                    # then it is a variable or method.
                    # Treat all methods and variables same.
                    token_program.append(var_token)
            else:
                # Add token only if it present in token_to_id. Otherwise replace
                # the token with UNK_TOKEN.
                if token_name in token_to_id:
                    token_program.append(token_name)
                else:
                    token_program.append(unk_token)

        data[program_id]['tokens'] = token_program

    return data, token_to_id


def hash_generator(token_to_id, tokens):
    """Generate hash for tokens in 'tokens' using 'token_to_id'.

    Args:
        token_to_id: dict. A dictionary which maps each token to a unique ID.
        tokens: list(str). A list of tokens.

    Returns:
        int. Hash value generated for tokens in 'tokens' using 'token_to_id'.
    """
    hash_val = 0
    n = len(tokens) - 1
    for x in tokens:
        hash_val += token_to_id[x] * (len(token_to_id) ** n)
        n -= 1
    return hash_val


def k_gram_hash_generator(token_program, token_to_id, K):
    """Generate all k-gram hashes for tokenized program.

    Args:
        token_program: list(str). A list of tokens.
        token_to_id: dict. A dictionary which maps each token to a unique ID.
        K: int. Model parameter 'K' of winnowing.

    Returns:
        list(int). k-gram hashes generated from 'token_program' using
        'token_to_id'.

    """
    generated_hashes = [
        hash_generator(token_to_id, token_program[i: i+K])
        for i in xrange(0, len(token_program) - K + 1)]
    return generated_hashes


def generate_k_gram_hashes(data, token_to_id, K):
    """Generate k-gram hashes for all programs in dataset.

    Args:
        data: dict. A dictionary containing training data.
        token_to_id: dict. A dictionary which maps each token to a unique ID.
        K: int. Model parameter 'K' of winnowing.

    Returns:
        dict. A dictionary containing training data with additional key
        'k_gram_hashes' for each data point, storing k-gram hashes corresponding
        to that data point.
    """
    for program_id in data:
        data[program_id]['k_gram_hashes'] = k_gram_hash_generator(
            data[program_id]['tokens'], token_to_id, K)
    return data


def get_fingerprint_from_hashes(k_gram_hashes, window_size):
    """Generate document fingerprint from k-gram hashes of given program.

    Args:
        k_gram_hashes: list(int). k-gram hashes of program from which
            fingerprint is to be obtained.
        window_size: int. Size of the window from which fingerprints are to be
            obtained.

    Returns:
        list(int). Fingerprint obtained from k-gram hashes over 'window_size' of
        window.
    """
    generated_fingerprint = set()
    for i in xrange(0, len(k_gram_hashes) - window_size + 1):
        window_hashes = k_gram_hashes[i: i + window_size]
        min_hash_index = i + min(
            xrange(window_size), key=window_hashes.__getitem__)
        min_hash = k_gram_hashes[min_hash_index]
        generated_fingerprint.add((min_hash, min_hash_index))

    return list(generated_fingerprint)


def generate_program_fingerprints(data, T, K):
    """Generate document fingerprints for all programs in entire dataset.

    Args:
        data: dict. A dictionary containing training data.
        T: int. Model parameter 'T' of winnowing.
        K: int. Model parameter 'K' of winnowing.

    Returns:
        dict. A dictionary containing training data with additional key
        'fingerprint' for each program which stores generated fingerprint
        of program.
    """
    window_size = T - K + 1
    for program_id in data:
        data[program_id]['fingerprint'] = get_fingerprint_from_hashes(
            data[program_id]['k_gram_hashes'], window_size)
    return data


def calc_jaccard_index(set_a, set_b):
    """Calculate jaccard's coefficient for two sets set_a and set_b.

    Jaccard index of two set is equal to:
        (no. of elements in intersection of two sets)
        _____________________________________________
        (no. of elements in union of two sets)

    Args:
        set_a: list(int). First set.
        set_b: list(int). Second set.

    Returns:
        float. Jaccard index of two sets.

    """
    small_set = set_a[:] if len(set_a) < len(set_b) else set_b[:]
    union_set = set_b[:] if len(set_a) < len(set_b) else set_a[:]
    for elem in union_set:
        if elem in small_set:
            small_set.remove(elem)
    union_set.extend(small_set)

    if union_set == []:
        return 0

    small_set = set_a[:] if len(set_a) < len(set_b) else set_b[:]
    intersection_set = []
    for elem in small_set:
        if elem in set_a and elem in set_b:
            intersection_set.append(elem)
            set_a.remove(elem)
            set_b.remove(elem)

    coeff = float(len(intersection_set)) / len(union_set)
    return coeff


def get_program_similarity(fingerprint_a, fingerprint_b):
    """Find similarity between fingerprint of two programs.

    Args:
        fingerprint_a: list((int, int)). Fingerprint of first data point.
        fingerprint_b: list((int, int)). Fingerprint of second data point.

    Returns:
        float. Similarity between first and second program.
    """
    set_a = [h for (h, _) in fingerprint_a]
    set_b = [h for (h, _) in fingerprint_b]
    return calc_jaccard_index(set_a, set_b)


def generate_top_similars(data, top):
    """Find 'top' nearest neighbours for all programs in dataset.

    Args:
        data: dict. A dictionary containing training data.
        top: int. No. of nearest neighbours to be identified.

    Returns:
        dict. Dictionary containing training data with additional attribute
        'top_similar' in each data point which contains nearest 'top' no. of
        data points for each data point.
    """
    for program_id_1 in data:
        overlaps = []
        for program_id_2 in data:
            overlap = get_program_similarity(
                data[program_id_1]['fingerprint'],
                data[program_id_2]['fingerprint'])
            overlaps.append((program_id_2, overlap))
        data[program_id_1]['top_similar'] = sorted(
            overlaps, key=lambda e: e[1], reverse=True)[:top]
    return data


def run_knn(data, top_neighbours):
    """Predict classes for each program in dataset using KNN.

    Args:
        data: dict. A dictionary containing training data.
        top_neighbours: int. No. of nearest neighbours to consider for KNN.

    Returns:
        float. Average no. of times a class has to appear in nearest neighbour
            for it to be correct prediction.
        list(int). IDs of data points which are missclassified by KNN.
    """
    data = generate_top_similars(data, top_neighbours)

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
        cnt = collections.Counter(nearest_classes)
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


class CodeClassifier(base.BaseClassifier):
    """A class for code classifier that uses supervised learning to match
    Python programs to an answer group. The classifier trains on programs
    that exploration editors have assigned to an answer group.
    """
    def __init__(self):
        super(CodeClassifier, self).__init__()
        self.data = None
        self.clf = None
        self.token_to_id = None
        self.T = None
        self.K = None
        self.occurrence = None
        self.class_to_answer_group_mapping = None
        self.top = None

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
            'class_to_answer_group_mapping': self.class_to_answer_group_mapping,
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

        data, token_to_id = tokenize_data(data)

        # No. of nearest neighbours to consider for classification using KNN.
        top = len(answer_group_to_class_mapping) / 2

        # GRID search for best value of T and K. T and K are model parameters
        # of winnowing algorithm.
        results = []
        for T in range(T_MIN, T_MAX):
            for K in range(K_MIN, T + 1):
                data = generate_k_gram_hashes(data, token_to_id, K)
                data = generate_program_fingerprints(data, T, K)
                occurrence, missclassified_points = run_knn(data, top)
                accuracy = (
                    len(data) - len(missclassified_points)) / float(len(data))
                results.append((accuracy, T, K))

        best_score = sorted(results, key=lambda e: e[0], reverse=True)[0]
        T = best_score[1]
        K = best_score[2]

        # Run KNN for best value of T and K.
        data = generate_k_gram_hashes(data, token_to_id, K)
        data = generate_program_fingerprints(data, T, K)
        occurrence, missclassified_points = run_knn(data, top)

        # Create a new dataset for missclassified points.
        programs = [data[pid]['source'] for pid in sorted(data.keys())]

        # Build vocabulary for programs in dataset. This vocabulary will be
        # used to generate Bag-of-Words vector for python programs.
        cv = sklearn_text.CountVectorizer(tokenizer=cv_tokenizer, min_df=5)
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
            'gamma': [0.001, 0.01, 1, 3, 7]
        }]

        fit_params = {
            'sample_weight': sample_weight
        }
        search = model_selection.GridSearchCV(
            svm.SVC(), param_grid, fit_params=fit_params)
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
