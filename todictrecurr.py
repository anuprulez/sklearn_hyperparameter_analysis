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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
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

class SerializeClass:

    @classmethod
    def __init__(self):
        """ Init method. """
        self.weights_file = "weights.h5"
        self.filetypes = ['str', 'float', 'bool', 'NoneType', 'int', 'ndarray', 'tuple']

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
    def recursive_dict(self, cls_object, recur_dict={}):
        recur_dict["class_name"] = cls_object.__class__.__name__
        if "__module__" in dir(cls_object):
            recur_dict["path"] = cls_object.__module__
        else:
            recur_dict["path"] = cls_object.__class__.__module__
            
        if "__getstate__" in dir(cls_object):
            cls_object_states = cls_object.__getstate__()
            type_name = type(cls_object_states).__name__
            if type_name not in self.filetypes:
                for key, val in cls_object_states.items():
                    if type(val).__name__ not in self.filetypes:
                        recur_dict[key] = dict()
                        if key == "estimators_":
                            if "shape" in dir(val):
                                recur_dict[key]["shape"] = val.shape
                                estimator = val[0][0]
                            else:
                               recur_dict[key]["shape"] = len(val)
                               estimator = val[0]
                            self.recursive_dict(estimator, recur_dict[key])
                        else:
                            self.recursive_dict(val, recur_dict[key])
                        if key == "estimators_features_":
                            recur_dict[key] = val
                    else:
                        if type(val).__name__ in ['tuple']:
                            try:
                                val_getstate = val.__getstate__()
                                recur_dict[key][val_getstate[0]] = val_getstate[1]
                                recur_dict[key]["class_name"] = recur_dict.__class__.__name__
                            except:
                                continue
                        else:
                            if key == "estimators_":
                                recur_dict[key] = dict()
                                if "shape" in dir(val):
                                    recur_dict[key]["shape"] = val.shape
                                    estimator = val[0][0]
                                else:
                                   recur_dict[key]["shape"] = len(val)
                                   estimator = val[0]
                                self.recursive_dict(estimator, recur_dict[key])
                            else:
                                recur_dict[key] = val
        return recur_dict

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(C=3.0, kernel='poly', degree=5)
        clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        clf = LinearRegression()
        clf = GaussianNB()
        clf = SGDClassifier(loss='log', learning_rate='optimal', alpha=0.0001)
        clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        
        #clf = RadiusNeighborsClassifier()
        #clf = GradientBoostingClassifier(n_estimators=1)
        #clf = ExtraTreeClassifier()
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #clf = DecisionTreeRegressor()
        #clf = ExtraTreeRegressor()
        #clf = GradientBoostingClassifier(n_estimators=1)
        #clf = SVR()
        #clf = AdaBoostClassifier()
        #clf = BaggingClassifier()
        #clf = ExtraTreesClassifier()
        classifier, X_test, y_test, X = self.train_model(clf)
        get_states = classifier.__getstate__()
        #print(dir(classifier))
        #print(classifier.__reduce__())
        #klass, args, state = classifier.__reduce__()
        #print(dir(klass))
        #print(klass)
        #print(args[0].__module__)
        recur_dict = self.recursive_dict(classifier)
        print(recur_dict)
        #print("  ")
        #print(classifier.__reduce_ex__())
        #print("  ")
        #print(get_states)
        '''get_states["class_name"] = classifier.__class__.__name__
        get_states["class_path"] = classifier.__module__
        print("Serializing...")
        self.convert_to_hdf5(get_states)'''
        return X_test, y_test, classifier


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python serialize_hdf5.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
