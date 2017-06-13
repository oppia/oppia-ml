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

# INSTRUCTIONS:
#
# Run this script from the oppia-ml root folder:
#   bash scripts/run_backend_tests.sh
#
# It runs all the (Python) backend tests, in parallel.
#
# =====================
# CUSTOMIZATION OPTIONS
# =====================
#
# (1) Append a test target to make the script run all tests in a given module
# or class, or run a particular test. For example, appending
#
#   --test_target='foo.bar.Baz'
#
# runs all tests in test class Baz in the foo/bar.py module, and appending
#
#   --test_target='foo.bar.Baz.quux'
#
# runs the test method quux in the test class Baz in the foo/bar.py module.
#
# (2) Append a test path to make the script run all tests in a given
# subdirectory. For example, appending
#
#   --test_path='core/services'
#
# runs all tests in the core/services/ directory.
#
# (3) Enable the verbose log by add the argument. It will display the outputs of
#   the tests being run.
#
#   --verbose or -v
#
# IMPORTANT: Only one of --test_path and --test_target should be specified.

if [ -z "$BASH_VERSION" ]
then
  echo ""
  echo "  Please run me using bash: "
  echo ""
  echo "     bash $0"
  echo ""
  return 1
fi

set -e
source $(dirname $0)/setup.sh || exit 1

# Install third party dependencies
bash scripts/install_third_party.sh

$PYTHON_CMD scripts/backend_tests.py $@

echo ''
echo 'Done!'
echo ''
