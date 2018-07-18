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

import vmconf

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

    Args:
        classifier_data: dict|list. The original classifier data which needs
            conversion of floats to strings.

    Returns:
        dict|list. Modified classifier data in which float values are converted
            into strings and each dict/subdict is augmented with a special key
            'float_values' which contains list of keys whose values have
            undergone the transformation.
    """
    # pylint: disable=too-many-branches
    if isinstance(classifier_data, dict):
        if vmconf.FLOAT_INDICATOR_KEY in classifier_data:
            raise Exception(
                'Classifier data already contains a %s key' %
                vmconf.FLOAT_INDICATOR_KEY)
        float_fields = []
        for k in classifier_data:
            if isinstance(classifier_data[k], basestring):
                # A float value must not be stored as a string in
                # classifier data.
                if re.match(r'^([-+]?\d+\.\d+)$', classifier_data[k]):
                    raise Exception(
                        'Error: Found a float value %s stored as string. Float '
                        'values should not be stored as strings.' % (
                            classifier_data[k]))
            elif isinstance(classifier_data[k], int):
                classifier_data[k] = classifier_data[k]
            elif isinstance(classifier_data[k], dict):
                classifier_data[k] = (
                    convert_float_numbers_to_string_in_classifier_data(
                        classifier_data[k]))
            elif isinstance(classifier_data[k], list):
                # Recursive call to list returns two values. One is the new
                # updated list and other is a boolean stating presence of
                # a float value in list.
                new_list, is_there_any_float = (
                    convert_float_numbers_to_string_in_classifier_data(
                        classifier_data[k]))
                if is_there_any_float:
                    float_fields.append(k)
                classifier_data[k] = new_list
            elif isinstance(classifier_data[k], float):
                classifier_data[k] = str(classifier_data[k])
                float_fields.append(k)
            else:
                raise Exception(
                    'Expected all classifier data dict values to be dicts, '
                    'lists, floats, integers or strings but received %s.' % (
                        type(classifier_data[k])))
        classifier_data[vmconf.FLOAT_INDICATOR_KEY] = float_fields
        return classifier_data
    elif isinstance(classifier_data, list):
        new_list = []
        is_there_any_float = False
        for item in classifier_data:
            if isinstance(item, list):
                # Recursive call to list returns two values. One is the new
                # updated list and other is a boolean stating presence of
                # a float value in list.
                ret_list, ret_bool = (
                    convert_float_numbers_to_string_in_classifier_data(
                        item))
                # Either float value is already discovered or it is present in
                # the nested list.
                is_there_any_float = is_there_any_float or ret_bool
                new_list.append(ret_list)
            elif isinstance(item, float):
                new_list.append(str(item))
                is_there_any_float = True
            elif isinstance(item, dict):
                new_list.append(
                    convert_float_numbers_to_string_in_classifier_data(item))
                # A nested dict may contain a float value. If it doesn't
                # then its FLOAT_INDICATOR_KEY will have an empty list
                # and it will not be modified by decoder.
                is_there_any_float = True
            elif isinstance(item, basestring):
                # A float value must not be stored as a string in
                # classifier data.
                if re.match(r'^([-+]?\d+\.\d+)$', item):
                    raise Exception(
                        'Error: Found a float value %s stored as string. Float '
                        'values should not be stored as strings.' % (item))
                new_list.append(item)
            elif isinstance(item, int):
                new_list.append(item)
            else:
                raise Exception(
                    'Expected list values to be either strings, floats, '
                    'lists, integers or dicts but received %s.' % (type(item)))
        return new_list, is_there_any_float
    else:
        raise Exception(
            'Expected all top-level classifier data objects to be lists or '
            'dicts but received %s.' % (type(classifier_data)))
