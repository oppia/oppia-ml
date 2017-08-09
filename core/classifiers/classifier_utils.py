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
    if type(support_vectors) is scipy.sparse.csr.csr_matrix:
        # Warning: this might result in really large list.
        support_vectors = support_vectors.toarray()
        dual_coef = dual_coef.toarray()

    return {
        'n_support': clf.__dict__['n_support_'].tolist(),
        'support_vectors': support_vectors.tolist(),
        'dual_coef': dual_coef.tolist(),
        'intercept': clf.__dict__['_intercept_'].tolist(),
        'classes': clf.__dict__['classes_'].tolist(),
    }