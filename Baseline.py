import pdb
import numpy as np
import itertools
from ReadQM8 import *
from Labels import *
from NeuralNet import *

def RunBaseline():
    print("Reading excitation values.")
    YFull = ReadData('E1-CC2') # Row Vector
    YTrain, YVal, YTest = DivideData(YFull, TrainFrac = 0.8, ValFrac = 0.0)
    YTrain = YTrain.transpose()
    YVal = YVal.transpose()
    YTest = YTest.transpose()

    AvgTest = np.mean(YTrain)
    AEs = abs(YTest - AvgTest)
    MAE = np.mean(AEs)
    print(MAE)

RunBaseline()