"""
Serialize the classifier object and trained model.
It creates a model file for storing learned parameters.
"""

import sys
import h5py
import time
import numpy as np
import json

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM, SVR
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, MultiTaskLasso, ElasticNet, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
import xgboost

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import importlib

import deserialize_class


class SerializeClass:

    @classmethod
    def __init__(self):
        """ Init method. """
        self.weights_file = "weights.h5"

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
        return classifier, X_test, y_test, X_train

    @classmethod
    def convert_to_hdf5(self, classifier_dict):
        """
        Convert the definition of a class to HDF5
        """
        with h5py.File(self.weights_file, 'w') as h5file:
            for dict_item, val in classifier_dict.items():
              #print(dict_item)
              if val is not None:
                  type_name = type(val).__name__
                  try:
                      #print(dict_item, val)
                      if dict_item == "estimators_":
                          print(val.shape)
                      if type_name in ['ndarray']:
                          h5file.create_dataset(dict_item, (val.shape), data=np.array(val, dtype=val.dtype.name))
                      else:
                          h5file.create_dataset(dict_item, data=val)
                  except:
                      if val:
                          if "__module__" in dir(val):
                              class_name = val.__class__.__name__
                              path = val.__module__
                              classkeys = val.__dict__
                              
                              dict_group = h5file.create_group(dict_item)
                              dict_group.create_dataset("class_name", data=class_name)
                              dict_group.create_dataset("path", data=path)
                              for item, item_val in classkeys.items():
                                  dict_group.create_dataset("attrs/" + item, data=item_val)
                          
                          '''if "__class__" in dir(val):
                              class_name = type(val).__name__
                              path = val.__class__.__module__
                              
                              dict_group = h5file.create_group(dict_item)
                              dict_group.create_dataset("class_name", data=class_name)
                              dict_group.create_dataset("path", data=path)
                              state_items = dict()
                              imp_attrs = [attr for attr in dir(val) if not attr.startswith("__") and not callable(getattr(val, attr))]
                              for key, value in val.__class__.__dict__.items():
                                  if key in imp_attrs:
                                      state_items[key] = eval("val." + key)
                              states = val.__getstate__()
                              for key, value in states.items():
                                  if key not in state_items:
                                      state_items[key] = value
                              for key, value in state_items.items():
                                  dict_group.create_dataset("attrs/" + key, data=value)'''
                          
                          if "data" in dir(val):
                              class_name = val.__class__.__name__
                              train_data = np.array(val.data)
                              dict_group = h5file.create_group(dict_item)
                              dict_group.create_dataset("class_name", data=class_name)
                              dict_group.create_dataset("data", (train_data.shape), data=np.array(train_data, dtype=train_data.dtype.name))
                      else:
                          h5file.create_dataset(dict_item, data=json.dumps(val))
                      continue

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(C=3.0, kernel='poly', degree=5)
        #clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        #clf = LinearRegression()
        clf = GaussianNB()
        clf = SGDClassifier(loss='log', learning_rate='optimal', alpha=0.0001)
        clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        
        #clf = RadiusNeighborsClassifier()
        clf = GradientBoostingClassifier(n_estimators=5)
        #clf = ExtraTreeClassifier()
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #clf = DecisionTreeRegressor()
        #clf = ExtraTreeRegressor()
        #clf = SVR()
        classifier, X_test, y_test, X = self.train_model(clf)
        print(classifier)
        classifier_dict = classifier.__dict__
        classifier_dict["class_path"] = classifier.__module__
        classifier_dict["class_name"] = classifier.__class__.__name__
        print(classifier_dict)
        print("Serializing...")
        self.convert_to_hdf5(classifier_dict)
        return X_test, y_test, classifier


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python serialize_hdf5.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    deserialize = deserialize_class.DeserializeClass(serialize_clf.weights_file)
    de_classifier = deserialize.deserialize_class()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
