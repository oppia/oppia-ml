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

"""Registry for classification algorithms/classifiers."""

import os
import pkgutil

import vmconf


class Registry(object):
    """Registry of all classifier classes."""

    # pylint: disable=fixme
    # TODO (prasanna08): Add unittest for algorithm registry when we have
    # classifier(s) to test it.

    # Dict mapping algorithm IDs to classifier classes.
    _classifier_classes = {}

    @classmethod
    def get_all_classifier_algorithm_ids(cls):
        """Retrieves a list of all classifier algorithm IDs.

        Returns:
            A list containing all the classifier algorithm IDs.
        """
        return [classifier_id
                for classifier_id in vmconf.ALGORITHM_IDS]

    @classmethod
    def _refresh(cls):
        """Refreshes the dict mapping algorithm IDs to instances of
        classifiers.
        """
        cls._classifier_classes.clear()

        all_classifier_ids = cls.get_all_classifier_algorithm_ids()

        # Assemble all paths to the classifiers.
        extension_paths = [
            os.path.join(vmconf.CLASSIFIERS_DIR, classifier_id)
            for classifier_id in all_classifier_ids]

        # Crawl the directories and add new classifier instances to the
        # registry.
        for loader, name, _ in pkgutil.iter_modules(path=extension_paths):
            module = loader.find_module(name).load_module(name)

            try:
                clazz = getattr(module, name)
            except AttributeError:
                continue

            ancestor_names = [
                base_class.__name__ for base_class in clazz.__bases__]
            if 'BaseClassifier' in ancestor_names:
                cls._classifier_classes[clazz.__name__] = clazz

    @classmethod
    def get_all_classifiers(cls):
        """Retrieves a list of instances of all classifiers.

        Returns:
            A list of instances of all the classification algorithms.
        """
        if not cls._classifier_classes:
            cls._refresh()
        return [clazz() for clazz in cls._classifier_classes.values()]

    @classmethod
    def get_classifier_by_algorithm_id(cls, classifier_algorithm_id):
        """Retrieves a classifier instance by its algorithm id.

        Refreshes once if the classifier is not found; subsequently, throws a
        KeyError.

        Args:
            classifier_algorithm_id: str. The ID of the classifier algorithm.

        Raises:
            KeyError: If the classifier is not found the first time.

        Returns:
            An instance of the classifier.
        """
        if classifier_algorithm_id not in cls._classifier_classes:
            cls._refresh()
        clazz = cls._classifier_classes[classifier_algorithm_id]
        return clazz()
