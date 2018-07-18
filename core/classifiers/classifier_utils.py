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

import re

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
    """Validates that incoming object contains unicode literal strings.

    Args:
        classifier_data: *. The trained classifier model data.

    Raises:
        Exception. If any of the strings in classifier data is not a unicode
            string.
    """
    if isinstance(classifier_data, dict):
        for k in classifier_data.keys():
            if isinstance(k, str):
                raise Exception('Expected %s to be unicode but found str.' % k)
            unicode_validator_for_classifier_data(classifier_data[k])
    if isinstance(classifier_data, list):
        for item in classifier_data:
            unicode_validator_for_classifier_data(item)
    if isinstance(classifier_data, str):
        raise Exception(
            'Expected \'%s\' to be unicode but found str.' % classifier_data)


def convert_float_numbers_to_string_in_classifier_data(classifier_data):
    """Converts all floating point numbers in classifier data to string.

    The following function iterates through entire classifier data and converts
    all float values to corresponding string values. At the same time, it also
    verifies that none of the existing string values are convertible to float
    values with the help of regex.

    Args:
        classifier_data: dict|list|string|int|float. The original classifier
            data which needs conversion of floats to strings.

    Raises:
        Exception. If any of the string values are convertible to float then
            an exception is raised to report the error. The classifier data
            must not include any string values which can be casted to float.

    Returns:
        dict|list|string|int|float. Modified classifier data in which float
            values are converted into strings and each dict/subdict is augmented
            with a special key 'float_values' which contains list of keys whose
            values have undergone the transformation.
    """
    if isinstance(classifier_data, dict):
        for k in classifier_data:
            classifier_data[k] = (
                convert_float_numbers_to_string_in_classifier_data(
                    classifier_data[k]))
        return classifier_data
    elif isinstance(classifier_data, list):
        new_list = []
        for item in classifier_data:
            new_list.append(
                convert_float_numbers_to_string_in_classifier_data(item))
        return new_list
    elif isinstance(classifier_data, float):
        return str(classifier_data)
    elif isinstance(classifier_data, basestring):
        if re.match(r'^([-+]?\d+\.\d+)$', classifier_data):
            # A float value must not be stored as a string in
            # classifier data.
            raise Exception(
                'Error: Found a float value %s stored as string. Float '
                'values should not be stored as strings.' % (
                    classifier_data))
        return classifier_data
    elif isinstance(classifier_data, int):
        return classifier_data
    else:
        raise Exception(
            'Expected all classifier data objects to be lists, dicts, floats, '
            'strings, integers but received %s.' % (type(classifier_data)))
