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

"""This module provides interface to communicate with Oppia remotely."""

import hashlib
import hmac
import json
import requests

from core.platform import platform_services
import utils
import vmconf

metadata_services = platform_services.Registry.import_metadata_services()


def _get_url():
    if vmconf.DEV_MODE:
        return vmconf.DEFAULT_COMMUNICATION_URL

    return vmconf.SERVER_COMMUNICATION_URL


def _get_port():
    if vmconf.DEV_MODE:
        return vmconf.DEFAULT_COMMUNICATION_PORT

    return vmconf.SERVER_COMMUNICATION_PORT


def _get_vm_id():
    if vmconf.DEV_MODE:
        return vmconf.DEFAULT_VM_ID

    # Get VMID dynamically from metadata. HMAC module does not
    # support unicode string. Hence we need to cast them to str.
    return str(metadata_services.get_metadata_param(
        vmconf.METADATA_VM_ID_PARAM_NAME))


def _get_shared_secret():
    if vmconf.DEV_MODE:
        return vmconf.DEFAULT_VM_SHARED_SECRET

    # Get shared secret dynamically from metadata. HMAC module does not
    # support unicode string. Hence we need to cast them to str.
    return str(metadata_services.get_metadata_param(
        vmconf.METADATA_SHARED_SECRET_PARAM_NAME))


def generate_signature(data):
    """Generates digital signature for given data.

    Args:
        data: dict. A dictionary data to be transferred over network.

    Returns:
        str. The digital signature generated from request data.
    """
    msg = json.dumps(data, sort_keys=True)
    key = _get_shared_secret()

    # Generate signature and return it.
    return hmac.new(key, msg, digestmod=hashlib.sha256).hexdigest()


def fetch_next_job_request():
    """Returns the next job request to be processed.

    Returns:
        dict. A dict retrieved remotely from database containing
        job request data.
    """

    request_url = "%s:%s/%s" % (
        _get_url(), _get_port(), vmconf.FETCH_NEXT_JOB_REQUEST_HANDLER)

    payload = {
        'vm_id': _get_vm_id(),
        'message': _get_vm_id(),
    }
    signature = generate_signature(payload['message'])
    payload['signature'] = signature
    data = {
        'payload': json.dumps(payload)
    }
    response = requests.post(request_url, data=data)
    return utils.parse_data_received_from_server(response.text)


def store_trained_classifier_model(job_result_dict):
    """Stores the result of processed job request.

    Args:
        job_result_dict: dict. A dictionary containing result of training
            of classifier.

    Returns:
        response: response object containing the server's response.

    Raises:
        Exception: job_result_dict is not of dict type.
        Exception: job_result_dict does not contain 'job_id' key.
        Exception: job_result_dict does not contain
            'classifier_data_with_stringified_floats' key.
    """

    # Make sure that job_result_dict is in proper foramt.
    if not isinstance(job_result_dict, dict):
        raise Exception('job_result_dict must be in dict format.')

    if 'job_id' not in job_result_dict:
        raise Exception('job_result_dict must contain \'job_id\'.')

    if 'classifier_data_with_stringified_floats' not in job_result_dict:
        raise Exception(
            'job_result_dict must contain '
            '\'classifier_data_with_stringified_floats\'.')

    payload = {
        'message': job_result_dict,
        'vm_id': _get_vm_id()
    }
    signature = generate_signature(payload['message'])
    payload['signature'] = signature
    data = {
        'payload': json.dumps(payload)
    }
    request_url = "%s:%s/%s" % (
        _get_url(), _get_port(), vmconf.STORE_TRAINED_CLASSIFIER_MODEL_HANDLER)
    response = requests.post(request_url, data=data)
    return response.status_code
