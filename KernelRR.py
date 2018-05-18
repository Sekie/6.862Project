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
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from ReadQM8 import *
from Labels import *
from NeuralNet import *

def DoGridSearch(X, Y):
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]
    gammas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    TunedParameters = [{'alpha': alphas, 'gamma': gammas}]

    print(alphas)
    print(gammas)

    clf = GridSearchCV(KernelRidge(kernel = 'rbf'), TunedParameters, cv = 10, scoring = 'neg_mean_absolute_error', verbose = True)
    clf.fit(X, Y)
    ScoreGrid = -(clf.cv_results_['mean_test_score'].reshape(len(alphas),len(gammas)))
    plt.imshow(ScoreGrid, cmap = 'rainbow')
    plt.ylabel('alpha')
    plt.xlabel('gamma')
    plt.title('Kernel Ridge Regression Parameter Optimization (MAE)')
    plt.colorbar()
    plt.show()
    return clf.best_estimator_.alpha, clf.best_estimator_.gamma

def RunKernel(XTrain, YTrain, XVal, YVal, XTest, YTest):
    print("Optimizing Kernel Ridge Regression Parameters")
    #BestAlpha, BestGamma = DoGridSearch(XTrain, YTrain.ravel())
    BestAlpha = 0.01
    BestGamma = 0.001
    KRR = KernelRidge(kernel='laplacian', gamma=BestGamma, alpha = BestAlpha)
    KRR.fit(XTrain, YTrain.ravel())

    YPredTrain = KRR.predict(XTrain)
    DiffYTrain = abs(YPredTrain - YTrain.ravel())
    print(sum(DiffYTrain) / float(len(DiffYTrain)))

    YPred = KRR.predict(XTest)
    DiffY = abs(YPred - YTest.ravel())
    MAEPredicted = sum(DiffY) / float(len(DiffY))
    print(BestAlpha, BestGamma)
    print(MAEPredicted)
    
    plt.scatter(YTest.tolist(), YPred.tolist(), c = 'red', s = 5)
    plt.plot(np.linspace(0, 0.5, 2), np.linspace(0, 0.5, 2))
    plt.ylabel('Predicted Excitation Energy (a.u.)')
    plt.xlabel('True Excitation Energy (a.u.)')
    plt.title('Kernel Ridge Regression (Laplacian) Learned Excitation Energies')
    plt.show()
#RunKernel()