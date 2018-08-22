"""
Deserialize the class object and trained model
"""

import sys
import h5py
import time
import json
import numpy as np


class DeserializeClass:

    @classmethod
    def __init__(self, weights_file, definition_file):
        """ Init method. """
        self.weights_file = weights_file
        self.definition_file = definition_file

    @classmethod
    def import_module(self, class_path, class_name):
        """
        Import a module dynamically
        """
        class_module = __import__(class_path, fromlist=[class_name])
        classifier = getattr(class_module, class_name)
        return classifier

    @classmethod
    def deserialize_class(self):
        """
        Recreate the model using the class definition and weights
        """
        print("Deserializing...")
        clf = dict()
        with open(self.definition_file, 'r') as clf_definition_file:
            clf = json.loads(clf_definition_file.read())
        parameters = clf["definition"]
        class_name = clf["class_name"]
        class_path = clf["class_path"]
        classifier = self.import_module(class_path, class_name)
        classifier_obj = classifier()
        for key, val in parameters.items():
            '''if type(val).__name__ == 'dict' and 'type' in val:
                nested_cls = self.import_module(val["module"], val["class_name"])
                nested_cls_obj = nested_cls()
                nested_keys = val["params"].split(",")
                h5file = h5py.File(self.weights_file, 'r')
                for nes_key in nested_keys:
                    data = h5file.get(nes_key)[()]
                    setattr(nested_cls_obj, nes_key, data)
                setattr(classifier_obj, key, nested_cls_obj)
            else:'''
            setattr(classifier_obj, key, val)

        h5file = h5py.File(self.weights_file, 'r')
        for key in h5file.keys():
            data = h5file.get(key)[()]
            setattr(classifier_obj, key, data)
        return classifier_obj
