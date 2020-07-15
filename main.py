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

"""Main process for training classifiers."""

# Preconfigure before starting main worker process.
# This step should be performed before importing any of the
# third party libraries.

import logging
import sys
import time

import vm_config
vm_config.configure()

# pylint: disable=wrong-import-position
from core.domain import job_services
import vmconf

def main():
    """Main process of VM."""
    try:
        job_data = job_services.get_next_job()
        if not job_data:
            logging.info('No pending job requests.')
            return
        frozen_model_proto = job_services.train_classifier(
            job_data['algorithm_id'], job_data['algorithm_version'],
            job_data['training_data'])

        if frozen_model_proto:
            status = job_services.store_job_result(
                job_data['job_id'], job_data['algorithm_id'],
                frozen_model_proto)

            if status != 200:
                logging.warning(
                    'Failed to store result of the job with \'%s\' job_id',
                    job_data['job_id'])
        return

    except KeyboardInterrupt:
        logging.info('Exiting')
        sys.exit(0)

    except Exception as e: # pylint: disable=broad-except
        # Log any exceptions that arises during processing of job.
        logging.error(e.message)

    finally:
        if vmconf.DEFAULT_WAITING_METHOD == vmconf.FIXED_TIME_WAITING:
            time.sleep(vmconf.FIXED_TIME_WAITING_PERIOD)

if __name__ == '__main__':
    while True:
        main()
