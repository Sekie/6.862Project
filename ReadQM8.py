import numpy as np
import CoulombMatrix as CoulMat
import copy

# This function reads the QM8 and QM9 geometry files and forms the XYZ matrices associated for each molecule.
# The XYZ matrices of all molecules is stored in a list and returned as the first output.
# The atomic symbol of all atoms in the molecule is stored in a list and returned as the second output.
# The largest matrix size is the third output (needed to pad matrices). 
def ReadQM8():
    QM8 = open('qm8.sdf', 'r')
    AllXYZ = []
    AllAtoms = []
    LargestDim = 0

    MoleculeCount = 10
    MoleculeSize = 0
    XYZMat = np.zeros((0, 3))
    Atoms = []

    for Line in QM8:
        if MoleculeCount < 0:
            MoleculeCount = MoleculeCount + 1
        if Line[:10] == ' OpenBabel':
            MoleculeCount = -2
        if MoleculeSize > 0:
            if MoleculeCount < MoleculeSize:
                Split = Line.split()
                XYZMat = np.vstack((XYZMat, np.array(np.array([float(Split[0]), float(Split[1]), float(Split[2])]))))
                Atoms.append(Split[3])
            if MoleculeCount == MoleculeSize:
                AllXYZ.append(XYZMat)
                AllAtoms.append(Atoms)
                XYZMat = np.zeros((0, 3))
                Atoms = []
                if MoleculeSize > LargestDim:
                    LargestDim = copy.copy(MoleculeSize)
                MoleculeSize = 0
            MoleculeCount = MoleculeCount + 1
        if MoleculeCount == 0:
            Split = Line.split()
            MoleculeSize = int(Split[0])
    
    return AllXYZ, AllAtoms, LargestDim

# Generates np.array that can be used as input data for ML packages.
# Takes a list of np.arrays, a list of list of characters, and largest dimension.
def GenerateData(AllXYZ, AllAtoms, MaxDim, repr = 'sorted'):
    if repr == 'sorted':
        Cs = [] # Holds the list of sorted matrices
        for XYZ, Atom in zip(AllXYZ, AllAtoms):
            C = CoulMat.SortCoulombMat(CoulMat.XYZToCoulomb(XYZ, CoulMat.AtomSymToZ(Atom))) # Sorted coulomb matrix.
            Cs.append(C)
        Cs = CoulMat.PadMatrices(Cs, MaxDim) # Pads all matrices
        X = CoulMat.MakeFeatureMatrix(Cs) # Make feature represention matrix.
        return X