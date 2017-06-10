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

"""Common utilities for test classes."""

import contextlib
import unittest

import responses

import vmconf


class TestBase(unittest.TestCase):
    """Base class for all tests."""

    def setUp(self):
        """setUp method which is run before every test case."""
        pass

    def tearDown(self):
        """tearDown method which is run after executing every test case."""
        pass

    def save_new_job_request(
            self, job_id, algorithm_id, training_data=None):
        """Puts a new job request with job_id.

        Args:
            job_id: str. Unicode string representing ID of
                new job request.
            algorithm_id: str. Unicode string representing ID of classifier
                algorithm.
            training_data: dict. Contains training data used to train
                classifier.
        """
        request_url = '%s:%s/%s' % (
            vmconf.DEFAULT_COMMUNICATION_URL, vmconf.DEFAULT_COMMUNICATION_PORT,
            vmconf.FETCH_NEXT_JOB_REQUEST_HANDLER)
        response_data = {
            'job_id': job_id,
            'algorithm_id': algorithm_id,
            'training_data': training_data
        }
        return self.put_get_request(request_url, response_data, 200)

    @staticmethod
    def put_get_request(url, data, status_code, headers=None):
        """Puts a mock get request for given url.

        Args:
            url: str. URL on which request.get() is to be executed.
            data: dict. A dictionary containing response data.
            status_code: int. Status code of response.
        """
        response = responses.RequestsMock()
        response.add(
            response.GET, url, json=data, status=status_code,
            adding_headers=headers)
        return response

    def set_job_result_post_callback(self, callback):
        """Sets a callback for store job result post request.

        Args:
            callback: callable. This is called implicitly when
                request.post() is executed.
        """
        request_url = '%s:%s/%s' % (
            vmconf.DEFAULT_COMMUNICATION_URL, vmconf.DEFAULT_COMMUNICATION_PORT,
            vmconf.STORE_TRAINED_CLASSIFIER_MODEL_HANDLER)
        return self.set_post_callback(request_url, callback)

    @staticmethod
    def set_post_callback(url, callback):
        """Sets a callback for store job result post request.

        Args:
            url: str. URL on which requests.post() is executed.
            callback: callable. This is called implicitly when
                request.post() is executed.
        """
        response = responses.RequestsMock()
        response.add_callback(
            response.POST, url, callback=callback)
        return response

    @staticmethod
    @contextlib.contextmanager
    def swap(obj, attr, newvalue):
        """Swap an object's attribute value within the context of a
        'with' statement. The object can be anything that supports
        getattr and setattr, such as class instances, modules, ...

        Example usage:

            import math
            with self.swap(math, 'sqrt', lambda x: 42):
                print math.sqrt(16.0)  # prints 42
            print math.sqrt(16.0)  # prints 4 as expected.

        Note that this does not work directly for classmethods. In this case,
        you will need to import the 'types' module, as follows:

            import types
            with self.swap(
                SomePythonClass, 'some_classmethod',
                types.MethodType(new_classmethod, SomePythonClass)):

        NOTE: self.swap and other context managers that are created using
        contextlib.contextmanager use generators that yield exactly once. This
        means that you can only use them once after construction, otherwise,
        the generator will immediately raise StopIteration, and contextlib will
        raise a RuntimeError.
        """
        original = getattr(obj, attr)
        setattr(obj, attr, newvalue)
        try:
            yield
        finally:
            setattr(obj, attr, original)

    @staticmethod
    def callback(func):
        """Decorator for callback method.

        Use this function as decorator when you are defining your own
        callback for post request.

        Example usage:

            @test_utils.GenericTestBase.callback
            def _post_callback(request):
                # Your assertions and code.

                # This function should return str(data) to be
                # returned in response.

        NOTE: It is neccessary to use this decorator whenever defining callback
        method for post request.
        """

        def wrapper(request):
            """Wrapper class for python decorator."""
            # func(request) returns str(data) as response to post request.
            return (200, {}, func(request))
        return wrapper


if vmconf.PLATFORM == 'gce':
    GenericTestBase = TestBase
else:
    raise Exception('Invalid platform: expected one of [\'gce\']')
