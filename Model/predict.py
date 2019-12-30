# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import inspect
import pandas as pd
import numpy as np
from train_model import *
import pickle
import argparse

#data loading
class ProDataset(Dataset):
    """ load dataset."""
    
    # Initialize your validation data(csv file)
    def __init__(self):

        df = pd.read_csv("Dataset/validation.csv")
        self.lines = list(df["Reaction SMILES"])
        self.properties = list(df["Reaction_Subclass"])
        self.len = len(self.properties)

        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        return self.lines[index], self.properties[index]

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):

        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)

def parse_args():
    """Parses arguments from terminal"""
    
    parser = argparse.ArgumentParser(description="Predict validation data by trained model")
    
    parser.add_argument("--model", "-model",
                        help=("trained model path/name."), type=str, required=True)
    
    #parser.add_argument("--data_to_pred", "-d", 
                        #help="data path/name to be predicted.", type=str, required=True)
    
    parser.add_argument("--output", "-o", 
                        help="prediction result.", type=str, required=True)

    return parser.parse_args()
        
BATCH_SIZE = 2
  
def main():
    
    args = parse_args()

    LOG.info("load data")

    val_dataset = ProDataset()
    dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = torch.load(args.model)


    LOG.info("prediction data")
    # prediction process
    Pred_result = []
    
    for batch_idx,(lines,properties) in enumerate(dataloader, 1):
        input, seq_lengths, y = make_variables(lines, properties)
        # model.batch_size = input.size(0)
        model.batch_size = y.shape[0] 
        pred,wts = model(input)
        pred = torch.max(pred.data.cpu(),1)[1]
        Pred_result.append(pred.cpu().data.numpy())
        
    Pred_results = [j for i in  Pred_result for j in i.tolist()]
    
    pattern = {0:'2N', 1:'2Y', 2:'3N', 3:'3Y'}
    
    P = [pattern[x] if x in pattern else x for x in Pred_results]

    LOG.info("output result")
    # output
    results = pd.DataFrame(P)

    results.to_csv(args.output, index=None)
    
    
LOG = lg.get_logger(name="Prediction oen data by trained model")

if __name__ == "__main__":
    main()
