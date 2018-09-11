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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, BaggingRegressor, AdaBoostRegressor, \
    ExtraTreesRegressor
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
from deepdish import io
import jsonpickler


class SerializeClass:

    @classmethod
    def __init__(self):
        """ Init method. """
        self.model_file = "model.h5"

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
    def save_model(self, model):
        """
        Save the dictionary to hdf5 file
        """
        print(model)
        se_model = jsonpickler.dump(model)
        #print(se_model)
        print("--------------")
        h5file = h5py.File(self.model_file, 'w')
        def recursive_save_model(h5file_obj, dictionary):
            for model_key, model_value in dictionary.items():
                type_name = type(model_value).__name__
                try:
                    if type_name in ['ndarray']:
                        h5file_obj.create_dataset(model_key, (model_value.shape), data=model_value)
                    elif type_name in ['int', 'int32', 'int64', 'float', 'float32', 'float64', 'str', 'tuple', 'bool', 'list', 'None', 'NoneType']:
                        if model_key in ["_aslist_", "_keys_"]:
                            h5file_obj.create_dataset(model_key, data=json.dumps(model_value))
                        elif type_name in ['None', 'NoneType']:
                            h5file_obj.create_dataset(model_key, data=json.dumps(model_value))
                        else:
                            h5file_obj.create_dataset(model_key, data=model_value)
                    elif type_name in ['dict']:
                        dict_group = h5file_obj.create_group(model_key)
                        recursive_save_model(dict_group, model_value)
                except Exception as exp:
                    print(model_key, exp)
                    continue
        recursive_save_model(h5file, se_model)

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(degree=5)
        #clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        #clf = LinearRegression()
        clf = GaussianNB()
        #clf = SGDClassifier(loss='hinge', learning_rate='optimal', alpha=0.0001)
        #clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        #clf = RadiusNeighborsClassifier()
        #clf = GradientBoostingClassifier(n_estimators=1)
        #clf = ExtraTreeClassifier()
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #clf = DecisionTreeRegressor()
        #clf = ExtraTreeRegressor()
        #clf = GradientBoostingClassifier(n_estimators=1)
        
        #clf = SVR()
        #clf = AdaBoostClassifier()
        #clf = AdaBoostRegressor()
        #clf = BaggingClassifier()
        #clf = BaggingRegressor()
        #clf = ExtraTreesClassifier(n_estimators=1)
        #clf = ExtraTreesRegressor()
        #clf = RandomForestClassifier()
        classifier, X_test, y_test, X = self.train_model(clf)
        print("Serializing...")
        self.save_model(classifier)
        
        return X_test, y_test, classifier


class DeserializeClass:

    @classmethod
    def __init__(self, model_file):
        """ Init method. """
        self.model_file = model_file
        
    @classmethod
    def load_model(self):
        """
        Read the hdf5 file recursively
        """
        print("Deserializing...")
        model_obj = dict()
        h5file = h5py.File(self.model_file, 'r')
        def recursive_load_model(h5file_obj, model_obj):
            for key in h5file_obj.keys():
                if h5file_obj.get(key).__class__.__name__ == 'Group':
                    model_obj[key] = dict()
                    recursive_load_model(h5file_obj[key], model_obj[key])
                else:
                    try:
                        key_value = h5file_obj.get(key).value
                        if key in ["_aslist_", "_keys_"]:
                            model_obj[key] = json.loads(key_value)
                        elif key_value in ['null']:
                            model_obj[key] = json.loads(key_value)
                        else:
                            if type(key_value).__name__ in ['ndarray']:
                                model_obj[key] = key_value.tolist()
                            else:
                                model_obj[key] = key_value
                    except Exception as exp:
                        print(key, exp)
                        continue
            return model_obj
        reconstructed_model = recursive_load_model(h5file, model_obj)
        #print(reconstructed_model)
        unloaded_model = jsonpickler.load(reconstructed_model)
        print(unloaded_model)
        return unloaded_model
        

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python todictrecurr.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    #se_classifier = jsonpickler.dump(classifier)
    deserialize = DeserializeClass(serialize_clf.model_file)
    de_classifier = deserialize.load_model()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
