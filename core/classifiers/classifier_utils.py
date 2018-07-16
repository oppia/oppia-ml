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


FLOAT_INDICATOR_KEY = 'float_values'


def convert_float_numbers_to_string_in_classifier_data(classifier_data):
    if isinstance(classifier_data, dict):
        if FLOAT_INDICATOR_KEY in classifier_data:
            raise Exception(
                'Classifier data already contains a %s key' %
                FLOAT_INDICATOR_KEY)
        float_fields = []
        for k in classifier_data.keys():
            if isinstance(classifier_data[k], (basestring, int)):
                classifier_data[k] = classifier_data[k]
            elif isinstance(classifier_data[k], dict):
                classifier_data[k] = (
                    convert_float_numbers_to_string_in_classifier_data(
                        classifier_data[k]))
            elif isinstance(classifier_data[k], list):
                new_list, is_any_value_float = (
                    convert_float_numbers_to_string_in_classifier_data(
                        classifier_data[k]))
                if is_any_value_float:
                    float_fields.append(k)
                classifier_data[k] = new_list
            elif isinstance(classifier_data[k], float):
                classifier_data[k] = str(classifier_data[k])
                float_fields.append(k)
            else:
                raise Exception(
                    'Expected all classifier data dict values to be dicts, '
                    'lists, floats, integers or strings but received %s.' %(
                        type(classifier_data[k])))
        classifier_data[FLOAT_INDICATOR_KEY] = float_fields
        return classifier_data
    elif isinstance(classifier_data, list):
        new_list = []
        is_any_value_float = False
        for item in classifier_data:
            if isinstance(item, list):
                ret_list, ret_bool = (
                    convert_float_numbers_to_string_in_classifier_data(
                        item))
                is_any_value_float = is_any_value_float or ret_bool
                new_list.append(ret_list)
            elif isinstance(item, float):
                new_list.append(str(item))
                is_any_value_float = True
            elif isinstance(item, dict):
                new_list.append(
                    convert_float_numbers_to_string_in_classifier_data(item))
            elif isinstance(item, (basestring, int)):
                new_list.append(item)
            else:
                raise Exception(
                    'Expected list values to be either strings, floats, '
                    'lists, integers or dicts but received %s.' % (type(item)))
        return new_list, is_any_value_float
    else:
        raise Exception(
            'Expected all top-level classifier data objects to be lists or '
            'dicts but received %s.' % (type(classifier_data)))
