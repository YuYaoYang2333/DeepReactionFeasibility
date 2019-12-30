"""
    Tokenize reaction SMILES and build a vocabulary.(create voc)
    SMILES parser from <OpenSMILES parser on python> --- https://github.com/recisic/Smipar

"""

import Smipar
import argparse
import log as lg
import pandas as pd
import rdkit
from rdkit import Chem


def parse_args():
    """Parses arguments from terminal"""
    
    parser = argparse.ArgumentParser(description="Generate voc from Dataset.")

    parser.add_argument("--input_data", "-i",
                        help=("Yours reaction SMILES file."), type=str, required=True)
    
    parser.add_argument("--output_voc", "-o", 
                        help="Prefix to the output path.", type=str, required=True)
    
    return parser.parse_args()


def load_reaction_data(file):
    """load data from CSV file, column["Reaction SMILES"] are used to build vocabulary \
        this single SMILES are all canonicalized RDKIT SMILES \
        return to a list of Reaction SMILES"""
    
    dataframe = pd.read_csv(file)
    Reaction_SMILES = list(dataframe["Reaction SMILES"])
    
    return Reaction_SMILES


def tokenization(content_list):
    "split Reaction_SMILES into three parts and tokenize these three parts respectively"
    
    voc = set()
    for c in content_list:
        v = [c2 for c1 in c.split(".") for c2 in c1.split(">")]
        for smi in v:
            token = Smipar.parser_list(smi)
            voc.update(set(token))
    voc = list(voc)
    voc.append(">")
    voc.append(".")
    
    return voc

def write_to_file(voc, file):
    """Write voc to a file."""
    with open(file, 'w') as f:
        for token in voc:
            f.write(token + "\n")
            


def main():
    "Main function"
    
    args = parse_args()
    
    LOG.info("Loading data...")
    Reaction_SMILES = load_reaction_data(args.input_data)
    
    LOG.info("Tokenization...")
    voc = tokenization(Reaction_SMILES)
    
    LOG.info("Output to a VOC file")
    write_to_file(voc, args.output_voc)
    
LOG = lg.get_logger(name="Generate VOC file")

if __name__ == "__main__":
    main()
