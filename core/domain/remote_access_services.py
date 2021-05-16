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

import base64
import hashlib
import hmac
import json
import requests

from core.domain.proto import training_job_response_payload_pb2
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


def generate_signature(message, vm_id):
    """Generates digital signature for given message combined with vm_id.

    Args:
        message: bytes. Message string.
        vm_id: bytes. ID of the VM that trained the job.

    Returns:
        str. The digital signature generated from request data.
    """
    msg = b'%s|%s' % (base64.b64encode(message), vm_id)
    key = _get_shared_secret().encode()

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
        'vm_id': _get_vm_id().encode(encoding='utf-8'),
        'message': _get_vm_id().encode(encoding='utf-8'),
    }
    signature = generate_signature(payload['message'], payload['vm_id'])
    payload['signature'] = signature
    data = {
        'payload': json.dumps(payload)
    }
    response = requests.post(request_url, data=data)
    return utils.parse_data_received_from_server(response.text)


def store_trained_classifier_model(job_result):
    """Stores the result of processed job request.

    Args:
        job_result: TrainingJobResult. Domain object containing result of
            training of classifier along with job_id and algorithm_id.

    Returns:
        int. Status code of the response.
    """

    job_result.validate()
    payload = training_job_response_payload_pb2.TrainingJobResponsePayload()
    payload.job_result.CopyFrom(job_result.to_proto())
    payload.vm_id = _get_vm_id().encode(encoding='utf-8')
    message = payload.job_result.SerializeToString().encode(encoding='utf-8')
    signature = generate_signature(message, payload.vm_id)
    payload.signature = signature

    data = payload.SerializeToString()
    request_url = "%s:%s/%s" % (
        _get_url(), _get_port(), vmconf.STORE_TRAINED_CLASSIFIER_MODEL_HANDLER)
    response = requests.post(
        request_url, data=data,
        headers={'Content-Type': 'application/octet-stream'})
    return response.status_code
