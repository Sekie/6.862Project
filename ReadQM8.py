import numpy as np

def ReadQM8():
    QM8 = open('qm8.sdf', 'r')
    AllXYZ = []
    AllAtoms = []
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
                MoleculeSize = 0
            MoleculeCount = MoleculeCount + 1
        if MoleculeCount == 0:
            Split = Line.split()
            MoleculeSize = int(Split[0])
    
    return AllXYZ, AllAtoms
a, b = ReadQM8()

print(a[0])
print(b[0])
print(a[2])
print(b[2])