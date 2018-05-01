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

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.keys = ['loss', 'acc', 'val_loss', 'val_acc']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()

def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=0, verbose=True):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Define the optimization
    model.compile(loss='mean_absolute_error', optimizer=Adam(), metrics=['mae'])
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
        val_acc, val_loss = history.values['epoch_val_loss'][-1], history.values['epoch_val_loss'][-1]
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

def CrossValidateNN(XFull, YFull, layers, k, epoch = 1):
    AvgError = 0.0
    test_acc = 0.0
    for j in range(k):
        data_split = np.array_split(XFull, k, 1)
        labels_split = np.array_split(YFull, k, 1)
        datajC = np.zeros(shape = (np.size(XFull, 0), 0))
        labelsjC = np.zeros((1,0))
        for i in range(k):
            if i == j:
                continue
            else:
                datajC = np.concatenate((datajC, data_split[i]), axis = 1)
                labelsjC = np.concatenate((labelsjC, labels_split[i]), axis = 1)
        model, history, val_acc, test_acc = run_keras(datajC.T, labelsjC.T, None, None, data_split[j].T, labels_split[j].T, layers, epochs = epoch)
        AvgError = AvgError + test_acc
    return AvgError / float(k)

def DivideData(X, TrainFrac = 6.0 / 10.0, ValFrac = 2.0 / 10.0):
    NumPts = X.shape[1]
    XTrain = X[:,:int(NumPts * TrainFrac)]
    XVal = X[:,int(NumPts * TrainFrac):int(NumPts * TrainFrac + NumPts * ValFrac)]
    XTest = X[:,int(NumPts * TrainFrac + NumPts * ValFrac):]
    return XTrain, XVal, XTest

def RunNN():
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
    
    print("Starting neural network.")
    layers = [Dense(input_dim = Dim, units = 100, activation='sigmoid'),
              Dense(units = 400, activation='sigmoid'),
              Dense(units = 1, activation="sigmoid")]

    #run_keras(XTrain, YTrain, XVal, YVal, XTest, YTest, layers, epochs = 1)

    # ****** Test error versus training set size.
    # kfold = range(2,12)
    # xval = []
    # datasize = []
    # for k in range(2, 21):
    #     xval_error = CrossValidateNN(XFull, YFull, layers, k, epoch = 10)
    #     xval.append(xval_error)
    #     datasize.append(float(NumPts) * (1.0 - 1.0 / float(k)))
    # print(datasize)
    # print(xval)
    # plt.plot(datasize, xval)
    # plt.show()

    # ****** Validation error versus epoch
    epoches = [1, 10, 20, 50, 75, 100, 200, 500, 750, 1000, 2000, 5000, 7500, 10000]
    vals_acc = []
    for ep in epoches:
        model, history, val_acc, test_acc = run_keras(XTrain, YTrain, XVal, YVal, XTest, YTest, layers, epochs = ep)
        vals_acc.append(val_acc)
    print(epoches)
    print(vals_acc)
    plt.plot(epoches, vals_acc)
    plt.ylabel('Validation MAE')
    plt.xlabel('Epochs')
    plt.title('Epoch Optimization')
    plt.show()

RunNN()