"""
Serialize the classifier object and trained model. It creates
two files, one JSON for class definition and another for storing learned parameters
"""

import sys
import h5py
import time
import json
import numpy as np

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, MultiTaskLasso, ElasticNet, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier


import deserialize_class


class SerializeClass:

    @classmethod
    def __init__(self):
        """ Init method. """
        self.weights_file = "weights.h5"
        self.definition_file = "definition.json"

    @classmethod
    def compute_prediction_score(self, classifier, X_test, y_test):
        """
        Evaluate classifier
        """
        predictions = classifier.predict(X_test)
        match = [1 for x,y in zip(predictions, y_test) if x == y]
        prediction = len(match) / float(len(predictions))
        print("Prediction score: %.2f" % prediction)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        # Fit and return the classifier
        classifier.fit(X_train, y_train)
        self.compute_prediction_score(classifier, X_test, y_test)
        return classifier, X_test, y_test

    @classmethod
    def convert_to_hdf5(self, classifier_dict):
        """
        Convert the definition of a class to JSON
        """
        with h5py.File(self.weights_file, 'w') as h5file:
            for dict_item, val in classifier_dict.items():
              if val is not None:
                  type_name = type(val).__name__
                  print(dict_item, val)
                  if type_name in ['ndarray']:
                      h5file.create_dataset(dict_item, (val.shape), data=np.array(val, dtype=val.dtype.name))
                  else:
                      h5file.create_dataset(dict_item, data=val)

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(C=3.0, kernel='poly', degree=5)
        #clf = LinearSVC(loss='hinge', tol=0.001, C=1)
        #clf = GradientBoostingClassifier()
        #clf = DecisionTreeClassifier()
        #clf = KNeighborsClassifier(n_neighbors=3)
        #clf = LinearRegression()
        #clf = GaussianNB()
        clf = LinearSVC()
        classifier, X_test, y_test = self.train_model(clf)
        # Get the attributes of the class object
        classifier_dict = classifier.__dict__
        classifier_dict["class_path"] = classifier.__module__ 
        classifier_dict["class_name"] = classifier.__class__.__name__
        print("Serializing...")
        self.convert_to_hdf5(classifier_dict)
        return X_test, y_test


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python serialize_hdf5.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test = serialize_clf.serialize_class()
    deserialize = deserialize_class.DeserializeClass(serialize_clf.weights_file, serialize_clf.definition_file)
    de_classifier = deserialize.deserialize_class()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
