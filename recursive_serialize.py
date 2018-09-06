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
    def convert_obj_to_dict(self, obj, result={}):
        for key, val in obj.items():
            if "__getstate__" in dir(val):
                result[key] = dict()
                typename = type(val.__getstate__()).__name__
                if typename in ['tuple']:
                    val_getstate = val.__getstate__()
                    result[key][val_getstate[0]] = val_getstate[1]
                    result[key]["class_name"] = val.__class__.__name__
                else:
                    class_name = val.__class__.__name__
                    path = val.__class__.__module__
                    result[key]["class_name"] = class_name
                    result[key]["path"] = path
                    self.convert_obj_to_dict(val.__getstate__(), result[key])
            else:
                result[key] = val
        return result

    @classmethod
    def convert_to_hdf5(self, classifier_dict):
        """
        Convert the definition of a class to HDF5
        """
        with h5py.File(self.weights_file, 'w') as h5file:
            for dict_item, val in classifier_dict.items():
                if val is not None:
                    type_name = type(val).__name__
                    try:
                        if type_name in ['ndarray']:
                            h5file.create_dataset(dict_item, (val.shape), data=np.array(val, dtype=val.dtype.name))
                        else:
                            h5file.create_dataset(dict_item, data=val)
                    except:
                        if 'shape' in dir(val):
                            shape = val.shape
                            dict_group = h5file.create_group(dict_item)
                            dict_group.create_dataset("shape", data=shape)
                            estimator = val[0][0]
                            estimator_state = estimator.__getstate__()
                            estimator_dict = dict()
                            estimator_dict = self.convert_obj_to_dict(estimator_state)
                            print(estimator_dict)
                            for item, value in estimator_dict.items():
                                if type(value) is dict:
                                    for k, v in value.items():
                                        type_name = type(v).__name__
                                        dict_group.create_dataset('attrs/' + item + '/' + k, data=v)
                                else:
                                    if value is not None:
                                        dict_group.create_dataset(item, data=value)
                        elif val:
                            class_name = val.__class__.__name__
                            if "data" in dir(val):
                                train_data = np.array(val.data)
                                dict_group = h5file.create_group(dict_item)
                                dict_group.create_dataset("class_name", data=class_name)
                                dict_group.create_dataset("data", (train_data.shape), data=np.array(train_data, dtype=train_data.dtype.name))
                            elif "__getstate__" in dir(val):
                                getstates = val.__getstate__()
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
                                    dict_group.create_dataset("attrs/" + key, data=value)
                else:
                    h5file.create_dataset(dict_item, data=json.dumps(val))

    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        #clf = SVC(C=3.0, kernel='poly', degree=5)
        #clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        #clf = LinearRegression()
        #clf = GaussianNB()
        #clf = SGDClassifier(loss='log', learning_rate='optimal', alpha=0.0001)
        #clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        
        #clf = RadiusNeighborsClassifier()
        clf = GradientBoostingClassifier(n_estimators=1)
        #clf = ExtraTreeClassifier()
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #clf = DecisionTreeRegressor()
        #clf = ExtraTreeRegressor()
        #clf = SVR()
        classifier, X_test, y_test, X = self.train_model(clf)
        get_states = classifier.__getstate__()
        get_states["class_name"] = classifier.__class__.__name__
        get_states["class_path"] = classifier.__module__
        print("Serializing...")
        self.convert_to_hdf5(get_states)
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
        class_path = h5file.get("class_path").value
        classifier = self.import_module(class_path, class_name)
        classifier_obj = classifier()
        for key in h5file.keys():
            if h5file.get(key).__class__.__name__ == 'Group':
                if key + "/shape" in h5file:
                    print(key)
                    shape = h5file.get(key+'/shape').value
                    print(shape)
                    for item, value in h5file.get(key).items():
                        try:
                            setattr(classifier_obj, item, value.value)
                        except:
                            continue
                    for item, value in h5file.get(key + "/attrs").items():
                        #if type(value) is dict:
                        for k, v in h5file.get(key + "/attrs/" + item).items():
                            print(k, v)
                        #else:
                         
                elif key + "/data" in h5file:
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
                        obj_class.__setstate__(obj_dict)
                        setattr(classifier_obj, key, obj_class)
                    else:
                        setattr(classifier_obj, key, obj)
            else:
                data = h5file.get(key).value
                if key not in ["class_name", "class_path"]:
                   setattr(classifier_obj, key, data)
        print(classifier_obj)
        return classifier_obj

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python serialize_hdf5.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    deserialize = DeserializeClass(serialize_clf.weights_file)
    de_classifier = deserialize.deserialize_class()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
