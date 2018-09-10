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
        if "data" in dir(cls_object):
            recur_dict["data"] = np.array(cls_object.data)
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
                        elif key == "tree_":
                            state_items = dict()
                            imp_attrs = [attr for attr in dir(val) if not attr.startswith("__") and not callable(getattr(val, attr))]
                            for k, v in val.__class__.__dict__.items():
                                if k in imp_attrs:
                                    state_items[k] = eval("val." + k)
                            states = val.__getstate__()
                            for k, v in states.items():
                                if k not in state_items:
                                    state_items[k] = v
                            state_items["class_name"] = val.__class__.__name__
                            if "__module__" in dir(val):
                                state_items["path"] = val.__module__
                            else:
                                state_items["path"] = val.__class__.__module__
                            recur_dict[key] = state_items
                        elif key == "estimators_features_":
                            recur_dict[key] = val
                        else:
                            self.recursive_dict(val, recur_dict[key])
                    else:
                        if type(val).__name__ is 'tuple':
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
    def save_hdf5(self, dictionary):
        """
        Save the dictionary to hdf5 file
        """
        with h5py.File(self.weights_file, 'w') as h5file:
            def recursive_save(dictionary, h5file_obj=None):
                for key, value in dictionary.items():
                    type_name = type(value).__name__
                    if not type_name in ['None', 'NoneType']:
                        try:
                            if type_name in ['ndarray']:
                                if key == "nodes":
                                    h5file_obj.create_dataset(key, (value.shape), data=value)
                                else:
                                    h5file_obj.create_dataset(key, (value.shape), data=np.array(value, dtype=value.dtype.name))
                            elif type_name in ['int', 'int32', 'int64', 'float', 'float32', 'float64', 'str', 'tuple', 'bool']:
                                h5file_obj.create_dataset(key, data=value)
                            elif type_name in ['dict']:
                                dict_group = h5file_obj.create_group(key)
                                recursive_save(value, dict_group)
                        except:
                            continue
            recursive_save(dictionary, h5file)

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        #clf = SVC(C=3.0, kernel='poly', degree=5)
        #clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        #clf = LinearRegression()
        #clf = GaussianNB()
        clf = SGDClassifier(loss='hinge', learning_rate='optimal', alpha=0.0001)
        #clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        
        #clf = RadiusNeighborsClassifier()
        #clf = GradientBoostingClassifier(n_estimators=1)
        #clf = ExtraTreeClassifier()
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #clf = DecisionTreeRegressor()
        #clf = ExtraTreeRegressor()
        #clf = GradientBoostingClassifier(n_estimators=1)
        #clf = SVR()
        clf = AdaBoostClassifier()
        #clf = BaggingClassifier()
        #clf = ExtraTreesClassifier()
        classifier, X_test, y_test, X = self.train_model(clf)
        get_states = classifier.__getstate__()
        classifier_dict = self.recursive_dict(classifier)
        print("Serializing...")
        self.save_hdf5(classifier_dict)
        return X_test, y_test, classifier


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
        class_path = h5file.get("path").value
        classifier = self.import_module(class_path, class_name)
        classifier_obj = classifier()
        for key in h5file.keys():
            if h5file.get(key).__class__.__name__ == 'Group':
                class_name = h5file.get(key + "/class_name").value
                class_path = h5file.get(key + "/path").value
                new_object = self.import_module(class_path, class_name)
                if key + "/data" in h5file:
                    data = h5file.get(key+'/data').value
                    obj = new_object(data)
                    setattr(classifier_obj, key, obj)
                else:
                    if class_name == 'Tree':
                        obj_dict = dict()
                        for k, v in h5file.get(key).items():
                            obj_dict[k] = v.value
                        obj_class = new_object(obj_dict["n_features"], obj_dict["n_classes"],  obj_dict["n_outputs"])
                        obj_class.__setstate__(obj_dict)
                        setattr(classifier_obj, key, obj_class)
            else:
                value = h5file.get(key).value
                value_type = type(value).__name__
                if value_type in ['ndarray']:
                    setattr(classifier_obj, key, value)
                else:
                    setattr(classifier_obj, key, value)
        print(classifier_obj)
        return classifier_obj


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python todictrecurr.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    deserialize = DeserializeClass(serialize_clf.weights_file)
    de_classifier = deserialize.deserialize_class()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
