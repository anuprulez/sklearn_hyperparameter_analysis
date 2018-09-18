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
from xgboost import XGBClassifier

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import jsonpickler_2 as jsonpickler

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer


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
        return classifier, X_test, y_test
        
        
    @classmethod
    def get_pipeline(self):
        """
        Construct pipelines
        """
        #X, y = samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)
        # ANOVA SVM-C
        '''anova_filter = SelectKBest(f_regression, k=5)
        clf = svm.SVC(kernel='poly', degree=5)
        anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
        anova_svm.set_params(anova__k=10, svc__C=.1)
        return anova_svm'''
        
        #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        '''tuned_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svm = SVC()
        clf = GridSearchCV(svm, tuned_parameters)
        return clf'''
        # You can set the parameters using the names issued
        # For instance, fit using a k of 10 in the SelectKBest
        # and a parameter 'C' of the svm
        #anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
        #prediction = anova_svm.predict(X)
        #anova_svm.score(X, y)        

        # getting the selected features chosen by anova_filter
        #anova_svm.named_steps['anova'].get_support()
        # Another way to get selected features chosen by anova_filter
        #anova_svm.named_steps.anova.get_support()
        #print anova_svm, X, y
        #estimators = [('reduce_dim', PCA()), ('clf', SVC())]
        #return Pipeline(estimators)
        #return make_pipeline(Binarizer(), MultinomialNB())
        estimators = [('reduce_dim', PCA()), ('clf', SVC())]
        pipe = Pipeline(estimators)
        param_grid = dict(reduce_dim__n_components=[2, 5, 10], clf__C=[0.1, 10, 100])
        clf = GridSearchCV(pipe, param_grid=param_grid)
        return clf
        
    @classmethod
    def serialize_class(self):
        """
        Convert to hdf5
        """
        clf = SVC(C=3.0, kernel='poly', degree=5)
        #clf = SVR()
        #clf = LinearSVC(loss='hinge', tol=0.001, C=2.0)
        #clf = LinearRegression(fit_intercept=True, n_jobs=2)
        #clf = GaussianNB()
        #clf = SGDClassifier(loss='hinge', learning_rate='optimal', alpha=0.0001)
        clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', algorithm='ball_tree', leaf_size=32)
        #clf = RadiusNeighborsClassifier()
        #clf = GradientBoostingClassifier(n_estimators=10)
        #clf = ExtraTreeClassifier()
        #clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #clf = DecisionTreeRegressor()
        #clf = ExtraTreeRegressor()
        #clf = GradientBoostingClassifier(n_estimators=10)
        #clf = AdaBoostClassifier(n_estimators=100)
        #clf = AdaBoostRegressor(n_estimators=100)
        #clf = BaggingClassifier()
        #clf = BaggingRegressor()
        #clf = ExtraTreesClassifier(n_estimators=10)
        #clf = ExtraTreesRegressor()
        #clf = RandomForestClassifier(random_state=123, n_estimators=100)
        #clf = XGBClassifier()
        clf = self.get_pipeline()
        classifier, X_test, y_test = self.train_model(clf)
        print("Serializing...")
        se_model = self.save_model(classifier)
        return X_test, y_test, classifier
        
    @classmethod
    def create_dataset(self, file_obj, key, value):
        """
        Create dataset
        """
        try:
            file_obj.create_dataset(key, data=json.dumps(value))
        except:
            file_obj.create_dataset(key, data=value)

    @classmethod
    def save_model(self, model):
        """
        Save the dictionary to hdf5 file
        """
        se_model = jsonpickler.dumpc(model)
        h5file = h5py.File(self.model_file, 'w')

        # nested method for recursion
        def recursive_save_model(h5file_obj, dictionary):
            for model_key, model_value in dictionary.items():
                type_name = type(model_value).__name__
                try:
                    if type_name in ['list']:
                        if len(model_value) > 0:
                            list_obj = all(isinstance(x, dict) for x in model_value)
                            if list_obj is False:
                                self.create_dataset(h5file_obj, model_key, model_value)
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
                                        self.create_dataset(h5file_obj, model_key_item, model_item)
                        else:
                            self.create_dataset(h5file_obj, model_key, model_value)
                    elif type_name in ['dict']:
                        if model_key in h5file_obj:
                            recursive_save_model(h5file_obj[model_key], model_value)
                        else:
                            group = h5file_obj.create_group(model_key)
                            recursive_save_model(group, model_value)
                    else:
                        self.create_dataset(h5file_obj, model_key, model_value)
                except Exception:
                    continue
        recursive_save_model(h5file, se_model)


class DeserializeClass:

    @classmethod
    def __init__(self, model_file):
        """ Init method. """
        self.model_file = model_file
        
    @classmethod
    def restore_list_in_model(self, model_object):
        """
        Convert dict to list if there are numbers as keys
        """
        if isinstance(model_object, dict) is True:
            for key, value in model_object.items():
                if type(value).__name__ in ['dict']:
                    keys = value.keys()
                    all_keys_number = all(str.isnumeric(x) for x in keys)
                    if all_keys_number is True:
                        keys = [int(ky) for ky in keys]
                        model_object[key] = list()
                        for idx in range(len(keys)):
                            model_object[key].append(value[str(idx)])
                self.restore_list_in_model(value)
        elif isinstance(model_object, list) is True:
            for k, v in enumerate(model_object):
                self.restore_list_in_model(v)

    @classmethod
    def load_model(self):
        """
        Read the hdf5 file recursively
        """
        print("Deserializing...")
        reconstructed_model = dict()
        h5file = h5py.File(self.model_file, 'r')

        # nested method for recursion
        def recursive_load_model(h5file_obj, reconstructed_model, counter=0):
            for key in h5file_obj.keys():
                # recurse if the item is a group
                if h5file_obj.get(key).__class__.__name__ in ['Group']:
                    reconstructed_model[key] = dict()
                    recursive_load_model(h5file_obj[key], reconstructed_model[key])
                else:
                    try:
                        key_value = h5file_obj.get(key).value
                        reconstructed_model[key] = json.loads(key_value)
                    except Exception:
                        if type(key_value).__name__ in ['ndarray']:
                            reconstructed_model[key] = key_value.tolist()
                        else:
                            reconstructed_model[key] = key_value
                        continue
        recursive_load_model(h5file, reconstructed_model)
        print('Restoring the list structure...')
        self.restore_list_in_model(reconstructed_model)
        return jsonpickler.loadc(reconstructed_model)


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python todictrecurr.py")
        exit(1)
    start_time = time.time()
    serialize_clf = SerializeClass()
    X_test, y_test, classifier = serialize_clf.serialize_class()
    deserialize = DeserializeClass("model.h5")
    de_classifier = deserialize.load_model()
    serialize_clf.compute_prediction_score(de_classifier, X_test, y_test)
    end_time = time.time()
    print("Program finished in %s seconds" % str(end_time - start_time))
