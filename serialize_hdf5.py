"""
Serialize the classifier object and trained model
"""

import sys
import h5py
import time
import json
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SerializeToHDF5:

    @classmethod
    def __init__(self):
        """ Init method. """
        self.weights = "weights.h5"
        self.definition = "definition.json"

    @classmethod
    def train_model(self, classifier):
        """
        Get a trained model
        """
        # Loading the dataset
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        X = digits.images.reshape((n_samples, -1))
        y = digits.target

        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

        # Fit and return the classifier
        classifier.fit(X_train, y_train)
        return classifier

    @classmethod
    def convert_to_json(self, classifier_members, classifier_dict):
        """
        Convert the definition of a class to JSON
        """
        with open(self.definition, 'w') as definition:
            clf = dict()
            clf_definition = dict()
            for item in classifier_members:
                if item == 'class_name' or item == 'class_path':
                    clf[item] = classifier_dict[item]
                else:
                    clf_definition[item] = classifier_dict[item]
            clf['definition'] = clf_definition
            definition.write(json.dumps(clf))

    @classmethod
    def convert_to_hdf5(self, classifier_dict):
        """
        Convert the learned parameters of a class to HDF5
        """
        with h5py.File(self.weights, 'w') as h5file:
            for dict_item, val in classifier_dict.items():
                  type_name = type(val).__name__
                  # Store only arrays (for now)
                  if val is not None and type_name == 'ndarray':
                      #print(dict_item)
                      #print(val.shape)
                      dset = h5file.create_dataset(dict_item, (val.shape), data=np.array(val, dtype=val.dtype.name))

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(C=2.0)
        classifier = self.train_model(clf)
        print(clf)
        # Get the attributes of the class object
        print("")
        print("Set the datasets in h5 file")
        classifier_dict = classifier.__dict__
        classifier_members = [attr for attr in classifier_dict.keys() if not attr.startswith("_") and not attr.endswith('_')]
        classifier_members.append("class_name")
        classifier_members.append("class_path")
        classifier_dict["class_path"] = classifier.__module__ 
        classifier_dict["class_name"] = classifier.__class__.__name__
        self.convert_to_json(classifier_members, classifier_dict)
        self.convert_to_hdf5(classifier_dict)

    @classmethod
    def retrieve_from_hdf5(self):
        """
        Get learned parameters from hdf5 file
        """
        # Read the arrays
        print("")
        print("Retrieve datasets...")
        h5file = h5py.File(self.weights, 'r')
        for key in h5file.keys():
            data = h5file.get(key)
            #print(key)
            #print(data.shape)

    @classmethod
    def deserialize_class(self):
        """
        Recreate the model using the class definition and weights
        """
        clf = dict()
        with open(self.definition, 'r') as clf_definition_file:
            clf = json.loads(clf_definition_file.read())
        parameters = clf["definition"]
        class_name = clf["class_name"]
        class_path = clf["class_path"]
        class_module = __import__(class_path, fromlist=[class_name])
        classifier = getattr(class_module, class_name)
        classifier_cls = classifier()
        for key, val in parameters.items():
            setattr(classifier_cls, key, val)
        print(classifier_cls)


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python serialize_hdf5.py" )
        exit( 1 )
    start_time = time.time()
    serialize_clf = SerializeToHDF5()
    serialize_clf.serialize_class()
    serialize_clf.deserialize_class()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
