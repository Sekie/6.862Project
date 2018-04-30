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

from ReadQM8 import *
from Labels import *

def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=0, verbose=True):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Define the optimization
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    N = X_train.shape[0]
    # Pick batch size
    batch = 32 if N > 1000 else 1     # batch size
    history = LossHistory()
    # Fit the model
    if X_val is None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split,
                  callbacks=[history], verbose=verbose)
    else:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val),
                  callbacks=[history], verbose=verbose)
    # Evaluate the model on validation data, if any
    if X_val is not None or split > 0:
        val_acc, val_loss = history.values['epoch_val_acc'][-1], history.values['epoch_val_loss'][-1]
        print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_acc))
    else:
        val_acc = None
    # Evaluate the model on test data, if any
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, history, val_acc, test_acc

def DivideData(X, TrainFrac = 6.0 / 10.0, ValFrac = 2.0 / 10.0):
    NumPts = X.shape[1]
    XTrain = X[:,:int(NumPts * TrainFrac)]
    XVal = X[:,int(NumPts * TrainFrac):int(NumPts * TrainFrac + NumPts * ValFrac)]
    XTest = X[:,int(NumPts * TrainFrac + NumPts * ValFrac):]
    return XTrain, XVal, XTest

def RunNN():
    AllXYZ, AllAtoms, MaxDim = ReadQM8()
    XFull = GenerateData(AllXYZ, AllAtoms, MaxDim)
    Dim = XFull.shape[0]
    XTrain, XVal, XTest = DivideData(X)

    YFull = ReadData('E1-CC2')
    print(YFull)
    YTrain, YVal, YTest = DivideData(YFull)
    
    layers = [Dense(input_dim= Dim, units = 400, activation='sigmoid'),
              Dense(input_dim= Dim, units = 100, activation="sigmoid")]

    run_keras(XTrain, YTrain, XVal, YVal, XTest, YTest, layers, epochs = 1)

RunNN()