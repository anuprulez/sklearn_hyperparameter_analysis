"""
Deserialize the class object and trained model. Use the 
class name and its path to import it dynamically. Then, set all 
its parameters with their respective data
"""

import sys
import h5py
import importlib
import json


class DeserializeClass:

    @classmethod
    def __init__(self, weights_file):
        """ Init method. """
        self.weights_file = weights_file

    @classmethod
    def import_module(self, class_path, class_name):
        """
        Import a module dynamically
        """
        module = importlib.import_module(class_path)
        classifier = getattr(module, class_name)
        print(classifier)
        return classifier

    @classmethod
    def deserialize_class(self):
        """
        Recreate the model using the class definition and weights
        """
        print("Deserializing...")
        h5file = h5py.File(self.weights_file, 'r')
        class_name = h5file.get("class_name").value
        class_path = h5file.get("class_path").value
        classifier = self.import_module(class_path, class_name)
        classifier_obj = classifier()
        for key in h5file.keys():
            if h5file.get(key).__class__.__name__ == 'Group':
                train_data = h5file.get(key+'/train_data').value
                class_name = h5file.get(key+'/class_name').value
                class_path_new = class_path.split('.')
                class_path_new = class_path_new[:len(class_path_new) - 1]
                class_path_new = ".".join(class_path_new)
                module_obj = self.import_module(class_path_new, class_name)
                val = module_obj(train_data)
                print(key, val)
                setattr(classifier_obj, key, val)
            else:
                data = h5file.get(key).value
                setattr(classifier_obj, key, data)
                print(key, data)
        return classifier_obj
