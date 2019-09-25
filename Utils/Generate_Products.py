# rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger.uncharge
# Using SMARTS to batch generate products

from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.six import StringIO
import re
sio = StringIO()
w = Chem.SDWriter(sio)

# load data
supp_alkyne = Chem.SDMolSupplier('../alkyne.sdf') # should change
supp_Br = Chem.SDMolSupplier('../Br.sdf') # should change
mol_alkyne = [m for m in supp_alkyne]
mol_Br = [m for m in supp_Br]

# reaction SMARTS
rxn = AllChem.ReactionFromSmarts('[N:2]1(:[C]:[C]([C]):[N:4]:[N:3]:1)[C].[*:8][C:5]#[C:6].[*:7][Br]>>[N:2]1([C:6]=[C:5]([*:8])[N:4]=[N:3]1)[*:7]')
rxn.GetNumReactantTemplates()
x1 = Chem.MolFromSmarts('[N]1(:[C]:[C]([C]):[N]:[N]:1)[C]')
# get products for data preproccess 
product_smiles = []
for i in range(len(mol_alkyne)):
    x2 = mol_alkyne[i]
    for j in range(len(mols_Br)):
        x3 = mol_Br[j]
        x = rxn.RunReactants((x1,x2,x3))
        c = x[0][0]
        smiles = Chem.MolToSmiles(c)
        product_smiles.append(smiles)
        
# name[i] and file path can be change (user-defined) Generate a single merge file
w=Chem.SDWriter("../products_all.sdf")
for i in range(len(product_smiles)):
    m=Chem.MolFromSmiles(product_smiles[i])
    #print(Chem.MolToMolBlock(m)) 
    m.SetProp("_Name",name[i])
    AllChem.Compute2DCoords(m)
    w.write(m)
w.close()

# Generate multiple independent sdf files
for i in range(len(product_smiles)):
    w=Chem.SDWriter("../products_spilt/" + name[i] + ".sdf")
    m=Chem.MolFromSmiles(product_smiles[i])
    #print(Chem.MolToMolBlock(m)) 
    m.SetProp("_Name",name[i])
    AllChem.Compute2DCoords(m)
    w.write(m)
w.close()
