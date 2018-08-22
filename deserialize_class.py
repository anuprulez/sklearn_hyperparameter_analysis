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
        class_module = __import__(class_path, fromlist=[class_name])
        classifier = getattr(class_module, class_name)
        classifier_cls = classifier()
        for key, val in parameters.items():
            setattr(classifier_cls, key, val)

        h5file = h5py.File(self.weights_file, 'r')
        for key in h5file.keys():
            data = h5file.get(key)[()]
            setattr(classifier_cls, key, data)
        return classifier_cls
