"""
Analyze hyperparameters of sklearn machine learning algorithms
"""

import sys
import numpy as np
import time
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# file paths
CURRENT_WORKING_DIR = os.getcwd()
DATA_DIR = CURRENT_WORKING_DIR + "/gradienttreeboosting/results/"


class AnalyseHyperparameters:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def read_file(self, path):
        """
        Read a file
        """
        return pd.read_csv(path, '\t')

    @classmethod
    def extract_rows(self, files, param1={}, param2={}, param3=[]):
        """
        Extract rows based on condition
        """
        lr = []
        estimators = []
        acc = []
        for file in files:
            file_content = self.read_file(join(DATA_DIR, file))
            corr_cols = file_content[(file_content[param1['name']] == param1['value']) & (file_content[param2['name']] == param2['value'])]
            lr.extend(corr_cols[param3[0]].tolist())
            estimators.extend(corr_cols[param3[1]].tolist())
            acc.extend(corr_cols[param3[2]].tolist())
        
        return lr, estimators, acc

    @classmethod
    def analyze_parameters(self, file_content, params_names_values={}, result_col_name=""):
        """
        Analyse the hyperparameters and compute correlation
        """
        files = [ file for file in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, file))]
        
        # find correlation between parameters 'param_estimator__learning_rate' and 'param_estimator__n_estimators'
        # keeping the other parameters same
        param1 = {'name': 'param_estimator__max_depth', 'value': 3}
        param2 = {'name': 'param_estimator__max_features', 'value': 'auto'}
        param3 = ["param_estimator__learning_rate", "param_estimator__n_estimators", "mean_test_score"]
        lr, estimators, acc = self.extract_rows(files, param1, param2, param3)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(lr, estimators, acc, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel("Learning rates", fontsize=14)
        ax.set_ylabel("No. of estimators", fontsize=14)
        ax.set_zlabel("Classification accuracy", fontsize=14)
        plt.title("Classification accuracy vs no. of estimators and learning rates (max depth: 3 and max features: auto)", fontsize=14)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(lr, estimators, acc, c='r', s=60)
        ax.set_xlabel("Learning rates", fontsize=14)
        ax.set_ylabel("No. of estimators", fontsize=14)
        ax.set_zlabel("Classification accuracy", fontsize=14)
        plt.title("Classification accuracy vs no. of estimators and learning rates (max depth: 3 and max features: auto)", fontsize=14)
        plt.show()


        # find correlation between parameters 'param_estimator__max_depth' and 'param_estimator__n_estimators'
        param1 = {'name': 'param_estimator__learning_rate', 'value': 0.1}
        param2 = {'name': 'param_estimator__max_features', 'value': 'auto'}
        param3 = ["param_estimator__max_depth", "param_estimator__n_estimators", "mean_test_score"]
        lr, estimators, acc = self.extract_rows(files, param1, param2, param3)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(lr, estimators, acc, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel("Max depth", fontsize=14)
        ax.set_ylabel("No. of estimators", fontsize=14)
        ax.set_zlabel("Classification accuracy", fontsize=14)
        plt.title("Classification accuracy vs max depth and no. of estimators( learning rate: 0.1 and max features: auto)", fontsize=14)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(lr, estimators, acc, c='r', s=60)
        ax.set_xlabel("Max depth", fontsize=14)
        ax.set_ylabel("No. of estimators", fontsize=14)
        ax.set_zlabel("Classification accuracy", fontsize=14)
        plt.title("Classification accuracy vs max depth and no. of estimators( learning rate: 0.1 and max features: auto)", fontsize=14)
        plt.show()


        # find correlation between parameters 'param_estimator__max_depth' and 'param_estimator__learning_rate'
        param1 = {'name': 'param_estimator__n_estimators', 'value': 100}
        param2 = {'name': 'param_estimator__max_features', 'value': 'auto'}
        param3 = ["param_estimator__max_depth", "param_estimator__learning_rate", "mean_test_score"]
        lr, estimators, acc = self.extract_rows(files, param1, param2, param3)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(lr, estimators, acc, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel("Max depth", fontsize=14)
        ax.set_ylabel("Learning rates", fontsize=14)
        ax.set_zlabel("Classification accuracy", fontsize=14)
        plt.title("Classification accuracy vs max depth and learning rates (n_estimators: 100 and max features: auto)", fontsize=14)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(lr, estimators, acc, c='r', s=60)
        ax.set_xlabel("Max depth", fontsize=14)
        ax.set_ylabel("Learning rates", fontsize=14)
        ax.set_zlabel("Classification accuracy", fontsize=14)
        plt.title("Classification accuracy vs max depth and learning rates (n_estimators: 100 and max features: auto)", fontsize=14)
        plt.show()


        

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python analyze_hyperparameters.py" )
        exit( 1 )
    start_time = time.time()
    res_column = "mean_test_score"
    params = {
        "param_estimator__learning_rate": [0.1, 0.05, 0.001, 0.005, 0.0001],
        "param_estimator__max_depth": [1, 2, 3, 4, 5],
        "param_estimator__max_features": ["None", "log2", "sqrt", "auto"],
        "param_estimator__n_estimators": [10, 50, 100, 200, 500]
    }
    parameters = AnalyseHyperparameters()
    parameters.analyze_parameters(params, res_column)
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
