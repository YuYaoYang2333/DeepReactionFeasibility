
# 2019/12/27
 
"""
    Using reaction SMARTS to generate batch products
"""


from __future__ import print_function
import argparse
import log as lg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.six import StringIO
import re
sio = StringIO()
w = Chem.SDWriter(sio)



def parse_args():
    """Parses arguments from terminal"""
    
    parser = argparse.ArgumentParser(description="Generate products from a batch of given reactants' SMILES.")

    parser.add_argument("--input_smiles_path_Br", "-i_Br",
                        help=("Yours Br SMILES file to run this script."), type=str, required=True)
    
    parser.add_argument("--input_smiles_path_Alkyne", "-i_Alkyne",
                        help=("Yours Alkyne SMILES file to run this script."), type=str, required=True)
    
    parser.add_argument("--output_products_types", "-t", 
                        help="Prefix to the output file type.", type=str, required=True) # [sdf/smi]
    
    parser.add_argument("--output_products_path", "-o", 
                        help="Prefix to the output products.", type=str, required=True)
    
    
    return parser.parse_args()


def load_data(smi_file):
    "Input: file including SMILES line by line \
        return a list(mol) "
    with open(smi_file, "r")as f:
        smis = f.read().splitlines()
    f.close()

    mols = [Chem.MolFromSmiles(i) for i in smis if Chem.MolFromSmiles(i)]
    return mols
    

# CuAAC reaction rxn pattern
pattern = '[N:2]1(:[C]:[C]([C]):[N:4]:[N:3]:1)[C].[*:8][C:5]#[C:6].[*:7][Br]>>[N:2]1([C:6]=[C:5]([*:8])[N:4]=[N:3]1)[*:7]'
rxn = AllChem.ReactionFromSmarts(pattern)


def Genetarion(mol_alkyne,mol_Br):
    "reaction SMARTS"
    
    rxn.GetNumReactantTemplates()
    reagent = Chem.MolFromSmarts('[N]1(:[C]:[C]([C]):[N]:[N]:1)[C]')
    products = []
    
    for i in range(len(mol_alkyne)):      
        reactant_a = mol_alkyne[i]
    
        for j in range(len(mol_Br)):       
            reactant_b = mol_Br[j]

            p = rxn.RunReactants((reagent,reactant_a,reactant_b))

            smi = Chem.MolToSmiles(p[0][0])
            products.append(smi)
        
    return products


def record_SDFfile(products, file):
    "written into a sdf file"
    w=Chem.SDWriter(file)
    
    for i in range(len(products)):
        m = Chem.MolFromSmiles(products[i])
        # m.SetProp("_Name",name[i]) # can set its name
        AllChem.Compute2DCoords(m)
        w.write(m)
    w.close()

    
def record_SMIfile(products, fileName):
    "written into a smi file"
    with open(fileName,'w+') as f:
        f.write("****************Products SMILES****************")
        f.write("\n")
        for i in products:
            f.write(i)
            f.write("\n")
        f.write("\n")
        f.write("****************" + str(len(products)) + "molecules have been written into this file" + "****************")
        f.write("\n")
        f.close()
    
    
def main():
    """Main function"""
    
    args = parse_args()


    LOG.info("Loading reactants data")
    mol_Br = load_data(args.input_smiles_path_Br)
    mol_alkyne = load_data(args.input_smiles_path_Alkyne)
    LOG.info("There are %d Br and %d Alkyne", len(mol_Br), len(mol_alkyne))
    

    LOG.info("Generate products......")
    products = Genetarion(mol_alkyne,mol_Br)
    LOG.info("Generate Done! There are %d molecules", len(products))
    
    
    LOG.info("Saving results into file", args.output_products_types)
    if args.output_products_types == "SDF":
        record_SDFfile(products, args.output_products_path)
        
    if args.output_products_types == "SMI":
        record_SMIfile(products, args.output_products_path)
        
    LOG.info("The End")
    
LOG = lg.get_logger(name="Generate products")

if __name__ == "__main__":
    main()
