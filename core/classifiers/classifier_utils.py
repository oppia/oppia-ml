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
        u'support_vectors': support_vectors,
        u'dual_coef': dual_coef,
        u'intercept': clf.__dict__['_intercept_'].tolist(),
        u'classes': clf.__dict__['classes_'].tolist(),
        u'probA': clf.__dict__['probA_'].tolist(),
        u'probB': clf.__dict__['probB_'].tolist(),
        u'kernel_params': kernel_params
    }


def unicode_validator_for_classifier_data(var):
    """Validates that incoming object contains unicode literal strings."""
    if isinstance(var, dict):
        for k in var.keys():
            if isinstance(k, str):
                raise Exception('Expected %s to be unicode but found str.' % k)
            unicode_validator_for_classifier_data(var[k])
    if isinstance(var, (list, set, tuple)):
        for item in var:
            unicode_validator_for_classifier_data(item)
    if isinstance(var, str):
        raise Exception('Expected \'%s\' to be unicode but found str.' % var)
