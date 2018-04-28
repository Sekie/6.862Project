import numpy as np
import CoulombMatrix as CM
import pandas as pd
from pyscf import gto, scf

def MolList(Atoms, Torsion):
    PTDataFrame = pd.read_csv("elements", skipinitialspace = True)
    PeriodicTable = dict(zip(PTDataFrame.Atomic_Number, PTDataFrame.Symbol))

    XYZ = CM.GenerateXYZ(Atoms, Torsion)
    print(XYZ)
    
    Molecule = []
    for i in range(len(Atoms)):
        if Atoms[i] == 0:
            continue
        Sym = str(PeriodicTable[Atoms[i]])
        AddAtom = [Sym, (XYZ[i, 0], XYZ[i, 1], XYZ[i, 2])]
        Molecule.append(AddAtom)
    return Molecule

# Here, we generate the atoms strings
def AllMolecules():
    AllAtomStrings = []
    for i in [6, 14]:
        for j in [1, 9, 17, 35]:
            AllAtomStrings.append([i, j, j, j, i, j, j, j])
    for i in [6, 14]:
        for j in [9, 17, 35]:
            AllAtomStrings.append([i, 1, j, j, i, 1, j, j])
    for i in [6, 14]:
        for j in [9, 17, 35]:
            AllAtomStrings.append([i, 1, 1, j, i, 1, 1, j])
    for i in [6, 14]:
        for j in [9, 17, 35]:
            AllAtomStrings.append([i, 1, 1, 1, i, 1, j, j])
    for i in [7]:
        for j in [1, 9, 17, 35]:
            AllAtomStrings.append([i, j, j, 0, i, j, j, 0])
    for i in [7]:
        for j in [9, 17, 35]:
            AllAtomStrings.append([i, 1, 1, 0, i, j, j, 0])
    for i in [8]:
        for j in [1, 9, 17, 35]:
            AllAtomStrings.append([i, j, 0, 0, i, j, 0, 0])

    Centers = [5, 6, 7, 8, 14, 15]
    Sub = [0, 1, 8, 9, 17, 35]
    for X in Centers:
        for Y in Centers:
            if Y < X:
                continue
            for A in Sub:
                for B in Sub:
                    if B < A:
                        continue
                    for C in Sub:
                        if C < B:
                            continue
                        for D in Sub:
                            for E in Sub:
                                if D < E:
                                    continue
                                for F in Sub:
                                    if F < E:
                                        continue
                                AllAtomStrings.append([X, A, B, C, Y, D, E, F])
                            

    # for X in range(1, 19):
    #     for Y in range(1, 19):
    #         for A in range(0, 19):
    #             for D in range(0, 19):
    #                 for E in range(D, 19):
    #                     for F in range(E, 19):
    #                         AllAtomStrings.append([X, A, A, A, Y, D, E, F])
    return AllAtomStrings


def RunPYSCF():
    Output = open('energies.txt', 'w')
    ListOfMolecules = AllMolecules()
    PESGridPoints = 10
    for Atoms in ListOfMolecules:
        for Grid in range(PESGridPoints):
            Torsion = 2 * np.pi * (Grid / PESGridPoints)
            Molecule = MolList(Atoms, Torsion)
            mol = gto.M(atom = Molecule, basis = '3-21g')
            mf = scf.RHF(mol)
            Energy = mf.kernel()
            Output.write(Energy)
