#!/usr/bin/env bash

# Copyright 2016 The Oppia Authors. All Rights Reserved.
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

set -e
source $(dirname $0)/setup.sh || exit 1

# Checking if pip is installed. If you are having
# trouble, please ensure that you have pip installed (see "Installing Oppia"
# on the Oppia developers' wiki page).
echo Checking if pip is installed on the local machine
if ! type pip > /dev/null 2>&1 ; then
    echo ""
    echo "  Pip is required to install Oppia-ml dependencies, but pip wasn't"
    echo "  found on your local machine."
    echo ""
    echo "  Please see \"Installing Oppia\" on the Oppia developers' wiki page:"

    if [ "${OS}" == "Darwin" ] ; then
      echo "    https://github.com/oppia/oppia/wiki/Installing-Oppia-%28Mac-OS%29"
    else
      echo "    https://github.com/oppia/oppia/wiki/Installing-Oppia-%28Linux%29"
    fi

    # If pip is not installed, quit.
    exit 1
fi

echo Installing third party libraries

# Install scikit-learn
SKLEARN_VERSION=0.18.1
SKLEARN_DIR=$THIRD_PARTY_DIR/sklearn-$SKLEARN_VERSION
echo Checking if scikit-learn is installed in $SKLEARN_DIR
if [ ! -d "$SKLEARN_DIR" ]; then
  echo Installing scikit-learn
  pip install scikit-learn==$SKLEARN_VERSION --target="$SKLEARN_DIR"
fi

# Install numpy
NUMPY_VERSION=1.12.1
NUMPY_DIR=$THIRD_PARTY_DIR/numpy-$NUMPY_VERSION
echo Checking if numpy is installed in $NUMPY_DIR
if [ ! -d "$NUMPY_DIR" ]; then
  echo Installing numpy
  pip install numpy==$NUMPY_VERSION --target="$NUMPY_DIR"
fi

# Install scipy
SCIPY_VERSION=0.19.0
SCIPY_DIR=$THIRD_PARTY_DIR/scipy-$SCIPY_VERSION
echo Checking if scipy is installed in $SCIPY_DIR
if [ ! -d "$SCIPY_DIR" ]; then
  echo Installing scipy
  pip install scipy==$SCIPY_VERSION --target="$SCIPY_DIR"
fi

# Install pylint
PYLINT_VERSION=1.7.1
PYLINT_DIR=$THIRD_PARTY_DIR/pylint-$PYLINT_VERSION
echo Checking if pylint is installed in $PYLINT_DIR
if [ ! -d "$PYLINT_DIR" ]; then
  echo Installing pylint
  pip install pylint==$PYLINT_VERSION --target="$PYLINT_DIR"
fi

# install pre-push script
echo Installing pre-push hook for git
$PYTHON_CMD $OPPIA_ML_DIR/scripts/pre_push_hook.py --install
