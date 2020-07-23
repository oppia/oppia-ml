#!/usr/bin/env bash

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

##########################################################################

# This script installs all necessary third party libraries in
# "third_party" folder.

set -e
source $(dirname $0)/setup.sh || exit 1

# Checking if pip is installed. If you are having
# trouble, please ensure that you have pip installed (see "Installing Oppia"
# on the Oppia developers' wiki page).
PIP_CMD="pip"
echo Checking if pip is installed on the local machine
if ! type $PIP_CMD > /dev/null 2>&1 ; then
  echo "Unable to find 'pip'. Trying pip2 instead..."
  PIP_CMD="pip2"
  if ! type $PIP_CMD > /dev/null 2>&1 ; then
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
fi

echo Installing third party libraries

while read -r line; do
  if [ "${line:0:1}" = '#' ] || [ "${#line}" = "0" ] ; then
    continue
  fi
  NAME=${line% *}
  VERSION=${line#* }
  LIB_PATH=$THIRD_PARTY_DIR/$NAME-$VERSION
  echo Checking if $NAME is installed in $LIB_PATH
  if [ ! -d "$LIB_PATH" ]; then
    echo Installing $NAME
    if [ $NAME == 'oppia-ml-proto' ]; then
      curl --output $THIRD_PARTY_DIR/tmp.zip "https://codeload.github.com/oppia/oppia-ml-proto/zip/$VERSION"
      unzip $THIRD_PARTY_DIR/tmp.zip -d $THIRD_PARTY_DIR
      rm $THIRD_PARTY_DIR/tmp.zip
      echo "DONE"
    else
      $PIP_CMD install $NAME==$VERSION --target="$LIB_PATH"
    fi
  fi
done < "$MANIFEST_FILE"

# install pre-push script
echo Installing pre-push hook for git
$PYTHON_CMD $OPPIA_ML_DIR/scripts/pre_push_hook.py --install
