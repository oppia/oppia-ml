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

"""Common utility functions required by classifier."""

import scipy

def extract_svm_parameters(clf):
    """Extract parameters from a trained SVC classifier.

    Args:
        clf: object of class sklearn.svm.SVC. Trained classifier model
             instance.

    Returns:
        dict. A dictionary containing parameters of trained classifier. These
        parameters will be used in frontend during prediction.
    """
    support_vectors = clf.__dict__['support_vectors_']
    dual_coef = clf.__dict__['_dual_coef_']

    # If `support_vectors` is a sparse matrix, convert it to an array.
    # Dual coefficients will have the same type as support_vectors.
    if isinstance(support_vectors, scipy.sparse.csr.csr_matrix):
        # Warning: this might result in really large list.
        support_vectors = support_vectors.toarray()
        dual_coef = dual_coef.toarray()

    kernel_params = {
        u'kernel': unicode(clf.__dict__['kernel']),
        u'gamma': clf.__dict__['_gamma'],
        u'coef0': clf.__dict__['coef0'],
        u'degree': clf.__dict__['degree'],
    }

    return {
        u'n_support': clf.__dict__['n_support_'].tolist(),
        u'support_vectors': support_vectors.tolist(),
        u'dual_coef': dual_coef.tolist(),
        u'intercept': clf.__dict__['_intercept_'].tolist(),
        u'classes': [i.decode('UTF-8') if isinstance(i, basestring) else i
                     for i in clf.__dict__['classes_'].tolist()],
        u'probA': clf.__dict__['probA_'].tolist(),
        u'probB': clf.__dict__['probB_'].tolist(),
        u'kernel_params': kernel_params
    }


def unicode_validator_for_classifier_data(classifier_data):
    """Validates that incoming object contains unicode literal strings."""
    if isinstance(classifier_data, dict):
        for k in classifier_data.keys():
            if isinstance(k, str):
                raise Exception('Expected %s to be unicode but found str.' % k)
            unicode_validator_for_classifier_data(classifier_data[k])
    if isinstance(classifier_data, (list, set, tuple)):
        for item in classifier_data:
            unicode_validator_for_classifier_data(item)
    if isinstance(classifier_data, str):
        raise Exception(
            'Expected \'%s\' to be unicode but found str.' % classifier_data)


def find_all_string_values_in_classifier_data(classifier_data, parent_key=''):
    """Finds all keys in classifier data which contain only string values."""
    join_key = lambda x, y: y if not x else '%s.%s' % (x, y)
    key_list = []
    if isinstance(classifier_data, dict):
        for k in classifier_data.keys():
            if isinstance(classifier_data[k], (str, unicode)):
                key_list.append(join_key(parent_key, k))
            else:
                ret_list = find_all_string_values_in_classifier_data(
                    classifier_data[k], k)
                if ret_list:
                    key_list += [join_key(parent_key, key) for key in ret_list]
        return key_list
    elif isinstance(classifier_data, (set, list, tuple)):
        all_values_are_string = True
        for item in classifier_data:
            if isinstance(item, (set, list, tuple)):
                ret_list = find_all_string_values_in_classifier_data(
                    item, parent_key)
                if not ret_list:
                    all_values_are_string = False
            elif not isinstance(item, (str, unicode)):
                all_values_are_string = False
            else:
                key_list += find_all_string_values_in_classifier_data(
                    item, parent_key)

        if all_values_are_string:
            return (
                [join_key(parent_key, key) for key in key_list] + [parent_key])
        return []
    return key_list


def convert_float_numbers_to_string_in_classifier_data(classifier_data):
    """Converts all floating point numbers in classifier data to string."""
    if isinstance(classifier_data, dict):
        for k in classifier_data.keys():
            if isinstance(classifier_data[k], float):
                classifier_data[k] = str(classifier_data[k])
            else:
                classifier_data[k] = (
                    convert_float_numbers_to_string_in_classifier_data(
                        classifier_data[k]))
        return classifier_data
    elif isinstance(classifier_data, (list, set, tuple)):
        new_list = []
        for item in classifier_data:
            if isinstance(item, float):
                new_list.append(str(item))
            else:
                new_list.append(
                    convert_float_numbers_to_string_in_classifier_data(
                        item))
        return type(classifier_data)(new_list)
    return classifier_data
