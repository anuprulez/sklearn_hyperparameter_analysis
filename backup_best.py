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
        print(classifier)
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
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(C=3.0, kernel='poly', degree=5)
        clf = SVR()
        clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        clf = LinearRegression(fit_intercept=True, n_jobs=2)
        clf = GaussianNB()
        clf = SGDClassifier(loss='hinge', learning_rate='optimal', alpha=0.0001)
        clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        #clf = RadiusNeighborsClassifier()
        clf = GradientBoostingClassifier(n_estimators=100)
        clf = ExtraTreeClassifier()
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf = DecisionTreeRegressor()
        # clf = ExtraTreeRegressor()
        #clf = GradientBoostingClassifier(n_estimators=10)
        clf = AdaBoostClassifier(n_estimators=2)
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
        
    @classmethod
    def save_model(self, model):
        """
        Save the dictionary to hdf5 file
        """
        se_model = jsonpickler.dump(model)
        print(se_model)
        h5file = h5py.File(self.model_file, 'w')
        def recursive_save_model(h5file_obj, dictionary):
            for model_key, model_value in dictionary.items():
                type_name = type(model_value).__name__
                try:
                    if type_name in ['ndarray']:
                        h5file_obj.create_dataset(model_key, (model_value.shape), data=model_value)
                    elif type_name in ['list']:
                        if len(model_value) > 0:
                            list_obj = all(isinstance(x, dict) for x in model_value)
                            if list_obj is False:
                                h5file_obj.create_dataset(model_key, data=json.dumps(model_value))
                            else:
                                for index, model_item in enumerate(model_value):
                                    model_key_item = model_key + "/" + str(index)
                                    if model_item is not None:
                                        if model_key_item in h5file_obj:
                                            recursive_save_model(model_key_item, model_item)
                                        else:
                                            group = h5file_obj.create_group(model_key_item)
                                            recursive_save_model(group, model_item)
                                    else:
                                        h5file_obj.create_dataset(model_key_item, data=json.dumps(model_item))
                        else:
                            h5file_obj.create_dataset(model_key, data=model_value)
                    elif type_name in ['int', 'int32', 'int64', 'float', 'float32', 'float64', 'str', 'tuple', 'bool', 'None', 'NoneType']:
                        if type_name in ['None', 'NoneType']:
                            h5file_obj.create_dataset(model_key, data=json.dumps(model_value))
                        else:
                            h5file_obj.create_dataset(model_key, data=model_value)
                    elif type_name in ['dict']:
                        if model_key in h5file_obj:
                            recursive_save_model(h5file_obj[model_key], model_value)
                        else:
                            group = h5file_obj.create_group(model_key)
                            recursive_save_model(group, model_value)
                except Exception as exp:
                    print(model_key, exp, model_value)
                    continue
        recursive_save_model(h5file, se_model)
        print("--------------")
        

class DeserializeClass:

    @classmethod
    def __init__(self, model_file):
        """ Init method. """
        self.model_file = model_file
        
    @classmethod
    def recurse_list_items(self, file_obj, list_dict):
        """
        Recurse list items
        """
        for key in file_obj.keys():
            if file_obj.get(key).__class__.__name__ == 'Group':
                if str.isnumeric(key) is True:
                    dict_key = list_dict
                else:
                    list_dict[key] = dict()
                    dict_key = list_dict[key]
                self.recurse_list_items(file_obj[key], dict_key)
            else:
                try:
                    key_value = file_obj.get(key).value
                    list_dict[key] = json.loads(key_value)
                except Exception as exp:
                    if type(key_value).__name__ in ['ndarray']:
                        list_dict[key] = key_value.tolist()
                    else:
                        list_dict[key] = key_value
                continue
        return list_dict
        
    @classmethod
    def load_model(self):
        """
        Read the hdf5 file recursively
        """
        print("Deserializing...")
        model_obj = dict()
        h5file = h5py.File(self.model_file, 'r')
        counter = 0
        list_dict = dict()
        def recursive_load_model(h5file_obj, model_obj, list_dict={}, counter=0):
            for key in h5file_obj.keys():
                if h5file_obj.get(key).__class__.__name__ == 'Group':
                    list_key = key + '/0'
                    if list_key in h5file_obj:
                        model_obj[key] = list()
                        while True:
                            list_key_iter = key + '/' + str(counter)
                            if list_key_iter in h5file_obj:
                                file_obj = h5file_obj[list_key_iter]
                                for recurse_key in file_obj.keys():
                                    if file_obj.get(recurse_key).__class__.__name__ == 'Group':
                                        model_obj[recurse_key] = dict()
                                        recursive_load_model(file_obj[recurse_key], model_obj[recurse_key], list_dict, counter)
                                    else:
                                        try:
                                            key_value = file_obj.get(recurse_key).value
                                            list_dict[recurse_key] = json.loads(key_value)
                                        except Exception as exp:
                                            if type(key_value).__name__ in ['ndarray']:
                                                list_dict[recurse_key] = key_value.tolist()
                                            else:
                                                list_dict[recurse_key] = key_value
                                            continue
                                
                                model_obj[key].append(list_dict)
                            else:
                                break
                            counter += 1
                            
                    else:
                        model_obj[key] = dict()
                        recursive_load_model(h5file_obj[key], model_obj[key])
                else:
                    model_obj[key] = dict()
                    try:
                        key_value = h5file_obj.get(key).value
                        model_obj[key] = json.loads(key_value)
                    except Exception as exp:
                        if type(key_value).__name__ in ['ndarray']:
                            model_obj[key] = key_value.tolist()
                        else:
                            model_obj[key] = key_value
                        continue
            return model_obj
        reconstructed_model = recursive_load_model(h5file, model_obj)
        print(reconstructed_model)
        #unloaded_model = jsonpickler.load(reconstructed_model)
        #return unloaded_model

    '''@classmethod
    def load_model(self):
        """
        Read the hdf5 file recursively
        """
        print("Deserializing...")
        model_obj = dict()
        h5file = h5py.File(self.model_file, 'r')
        def recursive_load_model(h5file_obj, model_obj, counter=0):
            for key in h5file_obj.keys():
                if h5file_obj.get(key).__class__.__name__ == 'Group':
                    list_key = key + '/0'
                    counter = 0
                    model_obj[key] = dict()
                    if list_key in h5file_obj:
                        list_dict = dict()
                        def recurse_list_items(file_obj, list_dict, counter):
                            model_obj[key] = list()
                            while True:
                                list_key_iter = key + '/' + str(counter)
                                if list_key_iter in h5file_obj:
                                    for recurse_key in file_obj.keys():
                                        if file_obj.get(recurse_key).__class__.__name__ == 'Group':
                                            #print(key, file_obj.get(recurse_key))
                                            if str.isnumeric(recurse_key) is True:
                                                dict_key = list_dict
                                                
                                            else:
                                                list_dict[recurse_key] = dict()
                                                dict_key = list_dict[recurse_key]
                                            list_dict[recurse_key] = dict()
                                            recursive_load_model(file_obj[recurse_key], list_dict, counter)
                                            #recurse_list_items(file_obj[recurse_key], dict_key)
                                        else:
                                            try:
                                                key_value = file_obj.get(recurse_key).value
                                                list_dict[recurse_key] = json.loads(key_value)
                                                #print(key, key_value, file_obj.get(recurse_key))
                                            except Exception as exp:
                                                if type(key_value).__name__ in ['ndarray']:
                                                    list_dict[recurse_key] = key_value.tolist()
                                                else:
                                                    list_dict[recurse_key] = key_value
                                                #print(recurse_key, key_value, file_obj.get(recurse_key))
                                                continue
                                else:
                                    break
                                counter += 1   
                        item_dict = recurse_list_items(h5file_obj[key], list_dict, counter)
                        model_obj[key].append(item_dict)
                    else:
                        recursive_load_model(h5file_obj[key], model_obj[key])
                else:
                    model_obj[key] = dict()
                    try:
                        key_value = h5file_obj.get(key).value
                        model_obj[key] = json.loads(key_value)
                    except Exception as exp:
                        if type(key_value).__name__ in ['ndarray']:
                            model_obj[key] = key_value.tolist()
                        else:
                            model_obj[key] = key_value
                        continue
            return model_obj
        reconstructed_model = recursive_load_model(h5file, model_obj)
        print(reconstructed_model)
        unloaded_model = jsonpickler.load(reconstructed_model)
        return unloaded_model'''


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python todictrecurr.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    se_classifier = jsonpickler.dump(classifier)
    deserialize = DeserializeClass(serialize_clf.model_file)
    de_classifier = deserialize.load_model()
    #serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))


