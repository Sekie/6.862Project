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

def DoGridSearch(X, Y):
    Cs = [1, 1e1, 1e2, 1e3, 1e4]
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    TunedParameters = [{'C': Cs, 'gamma': gammas}]

    print(Cs)
    print(gammas)

    clf = GridSearchCV(svm.SVR(kernel = 'laplacian'), TunedParameters, cv = 10, scoring = 'neg_mean_absolute_error', verbose = True)
    clf.fit(X, Y)
    ScoreGrid = -(clf.cv_results_['mean_test_score'].reshape(len(Cs),len(gammas)))
    plt.imshow(ScoreGrid, cmap = 'rainbow')
    plt.ylabel('C')
    plt.xlabel('gamma')
    plt.title('Kernel Ridge Regression Parameter Optimization (MAE)')
    plt.show()
    return clf.best_estimator_.C, clf.best_estimator_.gamma

def RunKernel(XTrain, YTrain, XVal, YVal, XTest, YTest):
    print("Optimizing Kernel Ridge Regression Parameters")
    BestC, BestGamma = DoGridSearch(XTrain, YTrain.ravel())
    # BestC = 100000.0
    # BestGamma = 1e-9
    KRR = svm.SVR(kernel='laplacian', degree=3, gamma=BestGamma, coef0=0.0, tol=0.001, C=BestC, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    KRR.fit(XTrain, YTrain.ravel())

    YPredTrain = KRR.predict(XTrain)
    DiffYTrain = abs(YPredTrain - YTrain.ravel())
    print(sum(DiffYTrain) / float(len(DiffYTrain)))

    YPred = KRR.predict(XTest)
    DiffY = abs(YPred - YTest.ravel())
    MAEPredicted = sum(DiffY) / float(len(DiffY))
    print(BestC, BestGamma)
    print(MAEPredicted)
    
    plt.scatter(YTest.tolist(), YPred.tolist(), c = 'red')
    plt.plot(np.linspace(0, 0.5, 2), np.linspace(0, 0.5, 2))
    plt.ylabel('Predicted Excitation Energy (eV)')
    plt.xlabel('True Excitation Energy (eV)')
    plt.title('Kernel Ridge Regression (rbf) Learned Excitation Energies')
    plt.show()
#RunKernel()