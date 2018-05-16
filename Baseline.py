import pdb
import numpy as np
import itertools
from ReadQM8 import *
from Labels import *
from NeuralNet import *

def RunBaseline(YTrain, YVal, YTest):
    AvgTest = np.mean(YTrain)
    AEs = abs(YTest - AvgTest)
    MAE = np.mean(AEs)
    print(MAE)