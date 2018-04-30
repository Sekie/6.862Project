import numpy as np
import pandas as pd
import copy

# Pads all matrices at once with zeros.
# Matrices - List of np.arrays 
# MaxDim - Int: maximum dimension of the matrices.
def PadMatrices(Matrices, MaxDim):
    PaddedMatrices = []
    for Matrix in Matrices:
        PadThai = np.zeros((MaxDim, MaxDim))
        PadThai[:Matrix.shape[0],:Matrix.shape[1]] = Matrix
        PaddedMatrices.append(PadThai)
    return PaddedMatrices

# Converts a list of element symbols (as characters) to a column vector of atomic numbers.
# Atom - List of characters
def AtomSymToZ(Atom):
    PTDataFrame = pd.read_csv("elements", skipinitialspace = True)
    PeriodicTable = dict(zip(PTDataFrame.Symbol, PTDataFrame.Atomic_Number))

    AtomicNumbers = np.zeros((0,1))
    for a in Atom:
        AtomicNumbers = np.vstack((AtomicNumbers, np.array([np.array([float(PeriodicTable[a])])])))
    return AtomicNumbers

# Takes one coulomb matrix and returns the upper triangle. This is just one column in the total matrix.
# Ordered as:
# [ 1  2  4 ]
# [    3  5 ]
# [       6 ]
# CoulombMat - np.array: Coulomb matrix of interest
def CoulMatToFeature(CoulombMat):
    NumAtoms = CoulombMat.shape[0]
    Xi = np.zeros((0, 1))
    for i in range(NumAtoms):
        for j in range(0, i + 1):
            Xi = np.vstack((Xi, np.array([np.array([CoulombMat[j, i]])])))
    return Xi

# Takes a list of matrices and turns it into a matrix of coulmn vectors representing the matrix.
# AllCoulombMat - List of np.array: All the ordered coulomb matrices
def MakeFeatureMatrix(AllCoulombMat):
    X = CoulMatToFeature(AllCoulombMat[0])
    for i in range(1, len(AllCoulombMat)):
        X = np.hstack((X, CoulMatToFeature(AllCoulombMat[i])))
    return X

# Sorts indexing based on norm of the column. Very dumb algorithm.
def SortCoulombMat(CoulombMat):
    CurrentRow = 0
    for i in range(CurrentRow, CoulombMat.shape[0]):
        NormC = 0
        NextLargestColumn = i
        for j in range(i, CoulombMat.shape[0]):
            NewNorm = np.linalg.norm(CoulombMat[:, j])
            if NewNorm > NormC:
                NormC = copy.copy(NewNorm)
                NextLargestColumn = j
        CoulombMat[[i, NextLargestColumn], [i, NextLargestColumn]] = CoulombMat[[NextLargestColumn, i], [NextLargestColumn, i]] 
    
    return CoulombMat

# Turns an XYZ matrix and list of atomic numbers into a coulomb matrix.
# XYZ - First dimension is atom, second dimension is xyz
# Atom - Column vector
def XYZToCoulomb(XYZ, Atom):
    NumAtoms = len(Atom)
    CoulombMatrix = np.zeros((NumAtoms, NumAtoms))

    # Start constructing the matrix
    for i in range(NumAtoms):
        for j in range(NumAtoms):
            if i == j:
                CoulombMatrix[i, j] = 0.5 * Atom[i]**2.4
            else:
                # Calculate the R between the atoms:
                dXYZ = XYZ[i, :] - XYZ[j, :]
                dR = np.linalg.norm(dXYZ)
                CoulombMatrix[i, j] = Atom[i] * Atom[j] / dR
                CoulombMatrix[j, i] = CoulombMatrix[i, j]
    return CoulombMatrix

def GenerateXYZ(Atoms, Torsion, DimerDist = 1.2, SubDist = 0.9, UmbrellaAng = 19.5):
    NumAtoms = len(Atoms)
    XYZ = np.zeros((NumAtoms, 3))
    
    # First atom centered at zero.
    # Second atom to fourth atoms are the substituents.
    dXYSub = SubDist * np.cos(UmbrellaAng * np.pi / 180.0)
    dZSub  = SubDist * np.sin(UmbrellaAng * np.pi / 180.0)
    XYZ[1, 0] = dXYSub
    XYZ[1, 2] = -dZSub
    
    dXSub = np.cos((60.0) * np.pi / 180.0) * dXYSub
    dYSub = np.sin((60.0) * np.pi / 180.0) * dXYSub
    XYZ[2, 0] = -dXSub
    XYZ[2, 1] = dYSub
    XYZ[2, 2] = -dZSub
    XYZ[3, 0] = -dXSub
    XYZ[3, 1] = -dYSub
    XYZ[3, 2] = -dZSub

    XYZ[4, 2] = DimerDist
    
    XYZ[5, 0] = dXYSub * np.cos((Torsion) * np.pi / 180.0)
    XYZ[5, 1] = dXYSub * np.sin((Torsion) * np.pi / 180.0)
    XYZ[5, 2] = DimerDist + dZSub

    XYZ[6, 0] = dXYSub * np.cos((120. + Torsion) * np.pi / 180.0)
    XYZ[6, 1] = dXYSub * np.sin((120. + Torsion) * np.pi / 180.0)
    XYZ[6, 2] = DimerDist + dZSub

    XYZ[7, 0] = dXYSub * np.cos((240. + Torsion) * np.pi / 180.0)
    XYZ[7, 1] = dXYSub * np.sin((240. + Torsion) * np.pi / 180.0)
    XYZ[7, 2] = DimerDist + dZSub

    return XYZ
