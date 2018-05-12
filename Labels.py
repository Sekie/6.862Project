import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pdb
import itertools
import operator
import pandas as pd

def ReadData(CalcLabel):
    CalcDF = pd.read_csv("test.sdf.csv")
    return CalcDF[[CalcLabel]].as_matrix().transpose()