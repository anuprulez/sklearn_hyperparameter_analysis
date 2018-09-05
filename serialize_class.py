"""
Serialize the classifier object and trained model.
It creates a model file for storing learned parameters.
"""

import sys
import h5py
import time
import numpy as numpy
import json

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, MultiTaskLasso, ElasticNet, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import xgboost
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import importlib


import sys
import os
import pandas
import re
import _pickle as pickle
import warnings
import numpy as np
import xgboost
import scipy
import sklearn
import ast
from asteval import Interpreter, make_symbol_table
from sklearn import (cluster, decomposition, ensemble, feature_extraction, feature_selection,
                    gaussian_process, kernel_approximation, linear_model, metrics,
                    model_selection, naive_bayes, neighbors, pipeline, preprocessing,
                    svm, linear_model, tree, discriminant_analysis)
import inspect


import deserialize_class



class SafePickler(object):
    """
    Used to safely deserialize scikit-learn model objects serialized by cPickle.dump
    Usage:
        eg.: SafePickler.load(pickled_file_object)
    """
    @classmethod
    def find_class(self, module, name):

        bad_names = ('and', 'as', 'assert', 'break', 'class', 'continue',
                    'def', 'del', 'elif', 'else', 'except', 'exec',
                    'finally', 'for', 'from', 'global', 'if', 'import',
                    'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
                    'raise', 'return', 'try', 'system', 'while', 'with',
                    'True', 'False', 'None', 'eval', 'execfile', '__import__',
                    '__package__', '__subclasses__', '__bases__', '__globals__',
                    '__code__', '__closure__', '__func__', '__self__', '__module__',
                    '__dict__', '__class__', '__call__', '__get__',
                    '__getattribute__', '__subclasshook__', '__new__',
                    '__init__', 'func_globals', 'func_code', 'func_closure',
                    'im_class', 'im_func', 'im_self', 'gi_code', 'gi_frame',
                    '__asteval__', 'f_locals', '__mro__')
        good_names = ['copy_reg._reconstructor', '__builtin__.object']

        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            fullname = module + '.' + name
            if  (fullname in good_names)\
                or  (   (   module.startswith('sklearn.')
                            or module.startswith('xgboost.')
                            or module.startswith('skrebate.')
                            or module.startswith('numpy.')
                            or module == 'numpy'
                        )
                        and (name not in bad_names)
                    ) :
                # TODO: replace with a whitelist checker
                '''if fullname not in sk_whitelist['SK_NAMES'] + sk_whitelist['SKR_NAMES'] + sk_whitelist['XGB_NAMES'] + sk_whitelist['NUMPY_NAMES'] + good_names:
                    print("Warning: global %s is not in pickler whitelist yet and will loss support soon. Contact tool author or leave a message at github.com" % fullname)'''
                mod = sys.modules[module]
                return getattr(mod, name)

        raise pickle.UnpicklingError("global '%s' is forbidden" % fullname)

    @classmethod
    def load(self, file):
        obj = pickle.Unpickler(file)
        obj.find_global = self.find_class
        return obj.load()

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
    def to_dict(self, obj):
        """
        turn a model object, including trained, into a JSON-able dictionary
        """
        primitive_types = [bool, int, float, complex, str, bytearray]
        dtypes = [numpy.bool_, numpy.int_, numpy.intc, numpy.intp, numpy.int8, numpy.int16,
                numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                numpy.uint8, numpy.float_, numpy.float16, numpy.float32, numpy.float64,
                numpy.complex_, numpy.complex64, numpy.complex128]

        t = type(obj)
        if obj is None or t in primitive_types:
            return obj
        if t is list:
            newlist = []
            for e in obj:
                newlist.append( self.to_dict(e) )
            return newlist
        if t is tuple:
            return tuple(self.to_dict(list(obj)))
        if t is set:
            return set(self.to_dict(list(obj)))
        if t is dict:
            newdict = {}
            for k, v in obj.items():
                newdict[k] = self.to_dict(v)
            return newdict

        name = getattr(obj, '__name__', None)
        if name is None:
            name = obj.__class__.__name__

        module_name = getattr(obj, '__module__', None)
        if module_name is None:
            module_name = obj.__class__.__module__

        _attributes_ = getattr(obj, '__dict__', None)
        if t is numpy.ndarray:
            _attributes_ = obj.tolist()
        elif t in dtypes:
            _attributes_ = obj.item()

        _attributes_ = self.to_dict(_attributes_)
        retv = {}
        retv['_module_'] = module_name
        retv['_name_'] = name
        retv['_attributes_'] = _attributes_
        return retv

    @classmethod
    def from_dict(self, data):
        """
        Construct a model object from a dictionary generated by to_dict
        """
        primitive_types = [bool, int, float, complex, str, bytearray]

        t = type(data)
        if data is None or t in primitive_types:
            return data
        if t is list:
            newlist = []
            for e in data:
                newlist.append( self.from_dict(e) )
            return newlist
        if t is tuple:
            return tuple( self.from_dict(list(data)) )
        if t is set:
            return set( self.from_dict(list(data)) )
        if t is dict:
            module = data.get('_module_', None)
            name = data.get('_name_', None)

            if not module and not name:
                newdict = {}
                for k, v in data.items():
                    newdict[k] = self.from_dict(v)
                return newdict

            klass = SafePickler.find_class(module, name)
            attributes = data.get('_attributes_', None)
            attributes = self.from_dict( attributes )
            if attributes is None:
                return klass
            if module == 'numpy':
                if name == 'ndarray':
                    return numpy.array(attributes)
                else:
                    return klass(attributes)
            obj = klass()
            for k, v in attributes.items():
                setattr(obj, k, v)
            return obj
        
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
                      if val:
                          #print(dict_item, type(val).__name__, dir(val))
                          imp_attrs = [attr for attr in dir(val) if not attr.startswith("__") and not callable(getattr(val, attr))]
                          #print(imp_attrs)
                          #print(val)
                          '''for item, value in val.__class__.__dict__.items():
                              if item in imp_attrs:
                                  print(item, dir(value))'''
                          
                          if "__class__" in dir(val):
                              class_name = type(val).__name__
                              path = val.__class__.__module__
                              dict_group = h5file.create_group(dict_item)
                              dict_group.create_dataset("class_name", data=class_name)
                              dict_group.create_dataset("path", data=path)
                              for key, value in val.__class__.__dict__.items():
                                  if key in imp_attrs:
                                      print(key)
                                      dict_group.create_dataset("attrs/" + key, data=eval("val." + key))

                          if "__module__" in dir(val):
                              print(dict_item, val)
                              class_name = val.__class__.__name__
                              path = val.__module__
                              classkeys = val.__dict__
                              dict_group = h5file.create_group(dict_item)
                              dict_group.create_dataset("class_name", data=class_name)
                              dict_group.create_dataset("path", data=path)
                              for item, item_val in classkeys.items():
                                  
                                  dict_group.create_dataset("attrs/" + item, data=item_val)
                              
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
        #clf = GradientBoostingClassifier()
        clf = ExtraTreesClassifier()
        #clf = DecisionTreeClassifier()
        #anova_filter = SelectKBest(f_regression, k=5)
        '''clf = svm.SVC(kernel='linear')
        anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
        anova_svm.set_params(anova__k=10, svc__C=1)'''

        classifier, X_test, y_test, X = self.train_model(clf)
        print(classifier)
        '''json_cls = self.to_dict(classifier)
        print(json_cls)
        cls = serialize_clf.from_dict(json_cls)'''
        #clfr = cls()
        #self.compute_prediction_score(cls, X_test, y_test)
        classifier_dict = classifier.__dict__
        classifier_dict["class_path"] = classifier.__module__
        classifier_dict["class_name"] = classifier.__class__.__name__
        print("Serializing...")
        self.convert_to_hdf5(classifier_dict)
        return X_test, y_test, X


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python serialize_hdf5.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, X = serialize_clf.serialize_class()
    deserialize = deserialize_class.DeserializeClass(serialize_clf.weights_file)
    de_classifier = deserialize.deserialize_class()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
