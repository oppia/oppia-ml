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

# This script starts oppia-ml on a local machine.

if [ -z "$BASH_VERSION" ]
then
  echo ""
  echo "  Please run me using bash: "
  echo ""
  echo "     bash $0"
  echo ""
  exit 1
fi

set -e
source $(dirname $0)/setup.sh || exit 1

if [ "$SETUP_DONE" ]; then
  echo 'Environment setup completed.'
fi

# Install third party dependencies.
bash scripts/install_third_party.sh

echo Starting main worker process
$PYTHON_CMD main.py
