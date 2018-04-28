import numpy as np

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
                CoulombMatrix[i, j] = Atom[i]**2.4
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
