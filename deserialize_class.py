"""
Deserialize the class object and trained model. Use the 
class name and its path to import it dynamically. Then, set all 
its parameters with their respective data
"""

import h5py


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
        h5file = h5py.File(self.weights_file, 'r')
        class_name = h5file.get("class_name").value
        class_path = h5file.get("class_path").value
        classifier = self.import_module(class_path, class_name)
        classifier_obj = classifier()
        for key in h5file.keys():
            data = h5file.get(key).value
            setattr(classifier_obj, key, data)
        return classifier_obj
