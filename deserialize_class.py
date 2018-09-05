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
                if key + "/data" in h5file:
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
                elif key + "/path" in h5file:
                    class_name = h5file.get(key + "/class_name").value
                    class_path = h5file.get(key + "/path").value
                    obj = self.import_module(class_path, class_name)
                    obj_dict = dict()
                    for item, value in h5file.get(key + "/attrs").items():
                        if class_name == 'Tree':
                            obj_dict[item] = value.value
                        else:
                            setattr(obj, item, value.value)
                    if class_name == 'Tree':
                        obj_class = obj(obj_dict["n_features"], obj_dict["n_classes"],  obj_dict["n_outputs"])
                        print(dir(obj_class))
                        '''obj_class.max_depth = obj_dict["max_depth"]
                        obj_class.capacity = obj_dict["capacity"]
                        obj_class.max_n_classes = obj_dict["max_n_classes"]
                        obj_class.node_count = obj_dict["node_count"]'''
                        setattr(classifier_obj, key, obj_class)
                    else:
                        setattr(classifier_obj, key, obj)
            else:
                data = h5file.get(key).value
                if key not in ["class_name", "class_path"]:
                   setattr(classifier_obj, key, data)
        print(classifier_obj)
        return classifier_obj
