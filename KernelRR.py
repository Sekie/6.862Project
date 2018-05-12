import pdb
import numpy as np
import itertools

np.random.seed(0)
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras import backend as K
from keras.initializers import VarianceScaling
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from ReadQM8 import *
from Labels import *
from NeuralNet import *

def DoGridSearch(KRR, X, Y):
    gammas = [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    Cs = [1, 10, 100, 1000]
    TunedParameters = [{'C': Cs, 'gamma': gammas}]
    clf = GridSearchCV(svm.SVR(kernel = 'rbf'), TunedParameters, cv = 10, scoring = 'neg_mean_absolute_error')
    clf.fit(X, Y)
    ScoreGrid = -(clf.cv_results_['mean_test_score'].reshape(len(Cs),len(gammas)))
    plt.imshow(ScoreGrid, cmap = 'rainbow', interpolation='nearest')
    plt.show()

def RunKernel():
    print("Preparing XYZ data.")
    AllXYZ, AllAtoms, MaxDim = ReadQM8()
    XFull = GenerateData(AllXYZ, AllAtoms, MaxDim) # Column vectors
    Dim = XFull.shape[0]
    NumPts = XFull.shape[1]
    XTrain, XVal, XTest = DivideData(XFull)
    XTrain = XTrain.transpose()
    XVal = XVal.transpose()
    XTest = XTest.transpose()

    print("Reading excitation values.")
    YFull = ReadData('E1-CC2') # Row Vector
    YTrain, YVal, YTest = DivideData(YFull)
    YTrain = YTrain.transpose()
    YVal = YVal.transpose()
    YTest = YTest.transpose()

    KRR = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    KRR.fit(XFull.transpose(), YFull.ravel())
    print("whew")
    
    DoGridSearch(KRR, XFull.transpose(), YFull.ravel())
RunKernel()