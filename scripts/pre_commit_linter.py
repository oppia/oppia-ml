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

"""Pre-commit script for Oppia-ml.

This script lints Python code, and prints a
list of lint errors to the terminal. If the directory path is passed,
it will lint all Python files in that directory; otherwise,
it will only lint files that have been touched in this commit.

IMPORTANT NOTES:

1.  Before running this script, you must install third-party dependencies by
    running

        bash scripts/start.sh

    at least once.

=====================
CUSTOMIZATION OPTIONS
=====================
1.  To lint only files that have been touched in this commit
        python scripts/pre_commit_linter.py

2.  To lint all files in the folder or to lint just a specific file
        python scripts/pre_commit_linter.py --path filepath

3.  To lint a specific list of files (*.py only). Separate files by spaces
        python scripts/pre_commit_linter.py --files file_1 file_2 ... file_n

Note that the root folder MUST be named 'oppia-ml'.
"""

# Pylint has issues with the import order of argparse.
# pylint: disable=wrong-import-order
import argparse
import fnmatch
import multiprocessing
import os
import re
import subprocess
import sys
import time
# pylint: enable=wrong-import-order

_PARSER = argparse.ArgumentParser()
_EXCLUSIVE_GROUP = _PARSER.add_mutually_exclusive_group()
_EXCLUSIVE_GROUP.add_argument(
    '--path',
    help='path to the directory with files to be linted',
    action='store')
_EXCLUSIVE_GROUP.add_argument(
    '--files',
    nargs='+',
    help='specific files to be linted. Space separated list',
    action='store')

BAD_PATTERNS = {
    '__author__': {
        'message': 'Please remove author tags from this file.',
        'excluded_files': (),
        'excluded_dirs': ()},
    'datetime.datetime.now()': {
        'message': 'Please use datetime.datetime.utcnow() instead of'
                   'datetime.datetime.now().',
        'excluded_files': (),
        'excluded_dirs': ()},
    '\t': {
        'message': 'Please use spaces instead of tabs.',
        'excluded_files': (),
        'excluded_dirs': (
            'assets/i18n/',)},
    '\r': {
        'message': 'Please make sure all files only have LF endings (no CRLF).',
        'excluded_files': (),
        'excluded_dirs': ()},
    'glyphicon': {
        'message': 'Please use equivalent material-icons '
                   'instead of glyphicons.',
        'excluded_files': (),
        'excluded_dirs': ()}
}

EXCLUDED_PATHS = (
    'third_party/*', '.git/*', '*.pyc', 'CHANGELOG',
    'scripts/pre_commit_linter.py')

if not os.getcwd().endswith('oppia-ml'):
    print ''
    print 'ERROR    Please run this script from the oppia root directory.'

_PYLINT_PATH = os.path.join(os.getcwd(), 'third_party', 'pylint-1.7.1')
_MANIFEST_FILE_PATH = os.path.join(os.getcwd(), 'manifest.txt')

if not os.path.exists(_PYLINT_PATH):
    print ''
    print 'ERROR    Please run start.sh first to install pylint '
    print '         and its dependencies.'
    sys.exit(1)

# Fix third-party library paths.
with open(_MANIFEST_FILE_PATH, 'r') as f:
    _PATHS_TO_INSERT = [
        os.path.join(
            os.getcwd(), 'third_party',
            '%s-%s' % (line.split()[0], line.split()[1])
        ) for line in [x.strip() for x in f.readlines()]
        if line and not line.startswith('#')
    ]

for path in _PATHS_TO_INSERT:
    sys.path.insert(0, path)

from pylint import lint  # pylint: disable=wrong-import-position

_MESSAGE_TYPE_SUCCESS = 'SUCCESS'
_MESSAGE_TYPE_FAILED = 'FAILED'


def _is_filename_excluded_for_bad_patterns_check(pattern, filename):
    """Checks if file is excluded from the bad patterns check.

    Args:
        pattern: str. The pattern to be checked against.
        filename: str. Name of the file.

    Returns:
        bool: Whether to exclude the given file from this
        particular pattern check.
    """
    return (any(filename.startswith(bad_pattern)
                for bad_pattern in BAD_PATTERNS[pattern]['excluded_dirs'])
            or filename in BAD_PATTERNS[pattern]['excluded_files'])


def _get_changed_filenames():
    """Returns a list of modified files (both staged and unstaged)

    Returns:
        a list of filenames of modified files.
    """
    unstaged_files = subprocess.check_output([
        'git', 'diff', '--name-only',
        '--diff-filter=ACM']).splitlines()
    staged_files = subprocess.check_output([
        'git', 'diff', '--cached', '--name-only',
        '--diff-filter=ACM']).splitlines()
    return unstaged_files + staged_files


def _get_all_files_in_directory(dir_path):
    """Recursively collects all files in directory and
    subdirectories of specified path.

    Args:
        dir_path: str. Path to the folder to be linted.
        excluded_glob_patterns: set(str). Set of all glob patterns
            to be excluded.

    Returns:
        a list of files in directory and subdirectories without excluded files.
    """
    files_in_directory = []
    for _dir, _, files in os.walk(dir_path):
        for file_name in files:
            filename = os.path.relpath(
                os.path.join(_dir, file_name), os.getcwd())

            files_in_directory.append(filename)
    return files_in_directory


def _lint_py_files(config_pylint, files_to_lint, result):
    """Prints a list of lint errors in the given list of Python files.

    Args:
        config_pylint: str. Path to the .pylintrc file.
        files_to_lint: list(str). A list of filepaths to lint.
        result: multiprocessing.Queue. A queue to put results of test.

    Returns:
        None
    """
    start_time = time.time()
    are_there_errors = False

    num_py_files = len(files_to_lint)
    if not files_to_lint:
        result.put('')
        print 'There are no Python files to lint.'
        return

    print 'Linting %s Python files' % num_py_files

    _BATCH_SIZE = 50
    current_batch_start_index = 0

    while current_batch_start_index < len(files_to_lint):
        # Note that this index is an exclusive upper bound -- i.e., the current
        # batch of files ranges from 'start_index' to 'end_index - 1'.
        current_batch_end_index = min(
            current_batch_start_index + _BATCH_SIZE, len(files_to_lint))
        current_files_to_lint = files_to_lint[
            current_batch_start_index : current_batch_end_index]
        print 'Linting Python files %s to %s...' % (
            current_batch_start_index + 1, current_batch_end_index)

        try:
            # This prints output to the console.
            lint.Run(current_files_to_lint + [config_pylint])
        except SystemExit as e:
            if str(e) != '0':
                are_there_errors = True

        current_batch_start_index = current_batch_end_index

    if are_there_errors:
        result.put('%s    Python linting failed' % _MESSAGE_TYPE_FAILED)
    else:
        result.put('%s   %s Python files linted (%.1f secs)' % (
            _MESSAGE_TYPE_SUCCESS, num_py_files, time.time() - start_time))

    print 'Python linting finished.'

def _get_all_files():
    """This function is used to check if this script is ran from
    root directory and to return a list of all the files for linting and
    pattern checks.
    """
    parsed_args = _PARSER.parse_args()
    if parsed_args.path:
        input_path = os.path.join(os.getcwd(), parsed_args.path)
        if not os.path.exists(input_path):
            print 'Could not locate file or directory %s. Exiting.' % input_path
            print '----------------------------------------'
            sys.exit(1)
        if os.path.isfile(input_path):
            all_files = [input_path]
        else:
            all_files = _get_all_files_in_directory(
                input_path)
    elif parsed_args.files:
        valid_filepaths = []
        invalid_filepaths = []
        for f in parsed_args.files:
            if os.path.isfile(f):
                valid_filepaths.append(f)
            else:
                invalid_filepaths.append(f)
        if invalid_filepaths:
            print ('The following file(s) do not exist: %s\n'
                   'Exiting.' % invalid_filepaths)
            sys.exit(1)
        all_files = valid_filepaths
    else:
        all_files = _get_changed_filenames()
    all_files = [
        filename for filename in all_files if not
        any(fnmatch.fnmatch(filename, pattern) for pattern in EXCLUDED_PATHS)]
    return all_files


def _pre_commit_linter(all_files):
    print 'Starting linter...'

    pylintrc_path = os.path.join(os.getcwd(), '.pylintrc')

    config_pylint = '--rcfile=%s' % pylintrc_path

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    py_files_to_lint = [
        filename for filename in all_files if filename.endswith('.py')]

    linting_processes = []

    py_result = multiprocessing.Queue()
    linting_processes.append(multiprocessing.Process(
        target=_lint_py_files,
        args=(config_pylint, py_files_to_lint, py_result)))

    print 'Starting Python Linting'
    print '----------------------------------------'

    for process in linting_processes:
        process.start()

    for process in linting_processes:
        # Require timeout parameter to prevent against endless waiting for the
        # linting function to return.
        process.join(timeout=600)

    print ''
    print '----------------------------------------'
    summary_messages = []

    # Require block = False to prevent unnecessary waiting for the process
    # output.
    summary_messages.append(py_result.get(block=False))
    print '\n'.join(summary_messages)
    print ''
    return summary_messages


def _check_newline_character(all_files):
    """This function is used to check that each file
    ends with a single newline character.
    """
    print 'Starting newline-at-EOF checks'
    print '----------------------------------------'
    total_files_checked = 0
    total_error_count = 0
    summary_messages = []
    all_files = [
        filename for filename in all_files if not
        any(fnmatch.fnmatch(filename, pattern) for pattern in EXCLUDED_PATHS)]
    failed = False
    for filename in all_files:
        with open(filename, 'rb') as f:
            total_files_checked += 1
            total_num_chars = 0
            for line in f:
                total_num_chars += len(line)
            if total_num_chars == 1:
                failed = True
                print '%s --> Error: Only one character in file' % filename
                total_error_count += 1
            elif total_num_chars > 1:
                f.seek(-2, 2)
                if not (f.read(1) != '\n' and f.read(1) == '\n'):
                    failed = True
                    print (
                        '%s --> Please ensure that this file ends'
                        'with exactly one newline char.' % filename)
                    total_error_count += 1

    if failed:
        summary_message = '%s   Newline character checks failed' % (
            _MESSAGE_TYPE_FAILED)
        summary_messages.append(summary_message)
    else:
        summary_message = '%s   Newline character checks passed' % (
            _MESSAGE_TYPE_SUCCESS)
        summary_messages.append(summary_message)

    print ''
    print '----------------------------------------'
    print ''
    if total_files_checked == 0:
        print 'There are no files to be checked.'
    else:
        print '(%s files checked, %s errors found)' % (
            total_files_checked, total_error_count)
        print summary_message

    return summary_messages


def _check_bad_patterns(all_files):
    """This function is used for detecting bad patterns.
    """
    print 'Starting Pattern Checks'
    print '----------------------------------------'
    total_files_checked = 0
    total_error_count = 0
    summary_messages = []
    all_files = [
        filename for filename in all_files if not
        any(fnmatch.fnmatch(filename, pattern) for pattern in EXCLUDED_PATHS)]
    failed = False
    for filename in all_files:
        with open(filename) as f:
            content = f.read()
            total_files_checked += 1
            for pattern in BAD_PATTERNS:
                if (pattern in content and
                        not _is_filename_excluded_for_bad_patterns_check(
                            pattern, filename)):
                    failed = True
                    print '%s --> %s' % (
                        filename, BAD_PATTERNS[pattern]['message'])
                    total_error_count += 1

    if failed:
        summary_message = '%s   Pattern checks failed' % _MESSAGE_TYPE_FAILED
        summary_messages.append(summary_message)
    else:
        summary_message = '%s   Pattern checks passed' % _MESSAGE_TYPE_SUCCESS
        summary_messages.append(summary_message)

    print ''
    print '----------------------------------------'
    print ''
    if total_files_checked == 0:
        print "There are no files to be checked."
    else:
        print '(%s files checked, %s errors found)' % (
            total_files_checked, total_error_count)
        print summary_message

    return summary_messages


def main():
    all_files = _get_all_files()
    newline_messages = _check_newline_character(all_files)
    linter_messages = _pre_commit_linter(all_files)
    pattern_messages = _check_bad_patterns(all_files)
    all_messages = linter_messages + newline_messages + pattern_messages
    if any([message.startswith(_MESSAGE_TYPE_FAILED) for message in
            all_messages]):
        sys.exit(1)


if __name__ == '__main__':
    main()
