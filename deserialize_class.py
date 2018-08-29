"""
Deserialize the class object and trained model. Use the 
class name and its path to import it dynamically. Then, set all 
its parameters with their respective data
"""

import sys
import h5py
import importlib
import json
import hickle
#import deepdish as dd
import hdf5_deepdish

class DeserializeClass:

    @classmethod
    def __init__(self, weights_file):
        """ Init method. """
        self.weights_file = weights_file
        self.weights_file_hickle = "classifier.hkl"

    @classmethod
    def import_module(self, class_path, class_name):
        """
        Import a module dynamically
        """
        module = importlib.import_module(class_path)
        classifier = getattr(module, class_name)
        return classifier

    @classmethod
    def deserialize_class(self):
        """
        Recreate the model using the class definition and weights
        """
        print("Deserializing...")
        #clf_obj = hickle.load(self.weights_file_hickle)
        clf_obj = hdf5_deepdish.load(self.weights_file)
        print(clf_obj)
        return clf_obj

        '''h5file = h5py.File(self.weights_file, 'r')
        class_name = h5file.get("class_name").value
        class_path = h5file.get("class_path").value
        classifier = self.import_module(class_path, class_name)
        classifier_obj = classifier()
        for key in h5file.keys():
            if h5file.get(key).__class__.__name__ == 'Group':
                train_data = h5file.get(key+'/data').value
                class_name = h5file.get(key+'/class_name').value
                class_path_modules = class_path.split('.')
                for index, item in enumerate(class_path_modules):
                    path = ".".join(class_path_modules[:len(class_path_modules) - index])
                    try:
                        module_obj = self.import_module(path, class_name)
                        val = module_obj(train_data)
                        setattr(classifier_obj, key, val)
                    except:
                        continue
            else:
                data = h5file.get(key).value
                setattr(classifier_obj, key, data)
        return classifier_obj'''
