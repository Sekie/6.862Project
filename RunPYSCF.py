import numpy as np
import CoulombMatrix as CM
import pandas as pd

def MolList(Atoms, Torsion):
    PTDataFrame = pd.read_csv("elements", skipinitialspace = True)
    PeriodicTable = dict(zip(PTDataFrame.Atomic_Number, PTDataFrame.Symbol))

    XYZ = CM.GenerateXYZ(Atoms, Torsion)
    print(XYZ)
    
    Molecule = []
    for i in range(len(Atoms)):
        Sym = str(PeriodicTable[Atoms[i]])
        AddAtom = [Sym, (XYZ[i, 0], XYZ[i, 1], XYZ[i, 2])]
        Molecule.append(AddAtom)
    return Molecule