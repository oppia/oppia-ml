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

"""Interface for platform services switching."""

import vmconf

class _Gce(object):
    """Provides platform-specific imports related to GCE
    (Google Compute Engine).
    """

    @classmethod
    def import_metadata_services(cls):
        """Imports and returns gce_metadata_services module.

        Returns:
            module. The gce_metadata_services module.
        """
        from core.platform.metadata import gce_metadata_services
        return gce_metadata_services

    NAME = 'gce'


class Registry(object):
    """Platform-agnostic interface for retrieving platform-specific modules.
    """

    # Maps platform names to the corresponding module registry classes.
    _PLATFORM_MAPPING = {
        _Gce.NAME: _Gce,
    }

    @classmethod
    def _get(cls):
        """Returns the appropriate interface class for platform-specific
        imports.

        Returns:
            class: The corresponding platform-specific interface class.
        """
        return cls._PLATFORM_MAPPING.get(vmconf.PLATFORM)

    @classmethod
    def import_metadata_services(cls):
        """Imports and returns metadata_services module.

        Returns:
            module. The metadata_services module.
        """
        return cls._get().import_metadata_services()
