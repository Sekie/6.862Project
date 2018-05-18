import numpy as np

from ReadQM8 import *
from Labels import *

def GetData():
    print("Preparing XYZ data.")
    AllXYZ, AllAtoms, MaxDim = ReadQM8()
    XFull = GenerateData(AllXYZ, AllAtoms, MaxDim) # Column vectors
    Dim = XFull.shape[0]
    NumPts = XFull.shape[1]
    
    print("Reading excitation values.")
    YFull = ReadData('E1-CC2') # Row Vector

    return XFull, YFull, Dim

def SplitData(XFull, YFull, Frac):
    XTrain, XVal, XTest = DivideData(XFull, TrainFrac = Frac, ValFrac = 0.9 - Frac)
    XTrain = XTrain.transpose()
    XVal = XVal.transpose()
    XTest = XTest.transpose()

    YTrain, YVal, YTest = DivideData(YFull, TrainFrac = Frac, ValFrac = 0.9 - Frac)
    YTrain = YTrain.transpose()
    YVal = YVal.transpose()
    YTest = YTest.transpose()

    print("Dataset Size: ", 21786 * Frac)
    print(XTrain.shape[0])
    return XTrain, YTrain, XTest, YTest