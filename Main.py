from NeuralNet import *
from KernelRR import *
from ReadQM8 import *

XTrain, YTrain, XVal, YVal, XTest, YTest, Dim = FormData(TrainFrac = 0.8, ValFrac = 0.1)
RunEpoch(XTrain, YTrain, XVal, YVal, XTest, YTest, Dim)