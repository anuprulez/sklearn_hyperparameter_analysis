"""
Serialize the classifier object and trained model
"""

import sys
import h5py
import time
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SerializeToHDF5:

    @classmethod
    def __init__(self):
        """ Init method. """
        self.file_path = "classifier.h5"

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
    def convert_to_hdf5(self):
        """
        Convert to hdf5
        """
        clf = SVC()
        classifier = self.train_model(clf)

        # Get the attributes of the class object
        print("")
        print("Set the datasets in h5 file")
        classifier_dict = classifier.__dict__
        with h5py.File(self.file_path, 'w') as h5file:
            for dict_item, val in classifier_dict.items():
                  type_name = type(val).__name__
                  # Store only arrays (for now)
                  if val is not None and type_name == 'ndarray':
                      print(dict_item)
                      print(val.shape)
                      dset = h5file.create_dataset(dict_item, (val.shape), data=np.array(val, dtype=val.dtype.name))
        self.retrieve_from_hdf5()

    @classmethod
    def retrieve_from_hdf5(self):
        """
        Get learned parameters from hdf5 file
        """
        # Read the arrays
        print("")
        print("Retrieve datasets...")
        h5file = h5py.File(self.file_path, 'r')
        for key in h5file.keys():
            data = h5file.get(key)
            print(key)
            print(data.shape)


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python serialize_hdf5.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = SerializeToHDF5()
    predict_tool.convert_to_hdf5()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
