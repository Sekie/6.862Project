from NeuralNet import *
from KernelRR import *
from ReadQM8 import *

XTrain, YTrain, XVal, YVal, XTest, YTest, Dim = FormData(TrainFrac = 0.9, ValFrac = 0.0)
RunKernel(XTrain, YTrain, XVal, YVal, XTest, YTest)