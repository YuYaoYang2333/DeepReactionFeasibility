# CuAAC_RFC : CuAAC Reaction Feasibility Classifier

## Introduction

This is a PyTorch implementation. The user can get feasibility judgments of click chemical reactions(CuAAC Reaction) by providing the reactants to be judged. We thank the previous work by Yoshua Bengio team. The code in this repository is based on their paper "A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING".


## Requirements

This package requires:
- Python 3.6.3
- PyTorch 1.1.0
- RDKit
- Numpy 1.14.2 
- Pandas 0.23.4 


## Usage

(1) You can generate products by given a batch of reactants SMILES:
```bash
bash generate_products.sh
```

(2) You should generate a vocabulary for training:
```bash
bash generate_voc.sh
```

(3) You can run a script for training:
```bash
bash train.sh
```

(4) You can run a script to predict external validation dataset using trained model(saved in pkl file):
```bash
bash predict.sh
```

Input Data file format: </br>
&nbsp;&nbsp;&nbsp;&nbsp;Datafile should be CSV file; </br>
&nbsp;&nbsp;&nbsp;&nbsp;The header must be "Reaction SMILES, Reaction Subclass(Can be filled with "2N"); </br>
&nbsp;&nbsp;&nbsp;&nbsp;Reaction SMILES = < Reactant_Br> ’.’ < Reactant_Alkyne > ’>’ < Product>


## Contact
Welcome to contact us. http://www.rcdd.org.cn/home/
