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
DATA_DIR = CURRENT_WORKING_DIR + "/results/estimator/"


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
    def analyze_parameters(self):
        """
        Analyse the hyperparameters and compute correlation
        """
        files = [ file for file in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, file))]
        print files
        NUM_COLORS = len(files)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        for index, file in enumerate(files):
            file_content = self.read_file(join(DATA_DIR, file))
            ax.scatter(file_content['param_estimator__n_estimators'], file_content['mean_test_score'])
        plt.legend(files)
        plt.grid(True)
        plt.show()
        

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python analyze_hyperparameters.py" )
        exit( 1 )
    start_time = time.time()
    parameters = AnalyseHyperparameters()
    parameters.analyze_parameters()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
