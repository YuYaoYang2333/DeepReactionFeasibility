
# coding: utf-8



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import inspect
import pandas as pd
import numpy as np
from train_over import *
import pickle

#data processing
class ProDataset(Dataset):
    """ load dataset."""
    
    # Initialize your data, download, etc.
    def __init__(self):
        df = val.copy()    
        self.lines = list(df["reaction_str"])
        self.properties = list(df["type"])
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

# load  data

data_file = sys.argv[1]
val = pd.read_csv(data_file)
val_dataset = ProDataset()
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
p_model = torch.load('./Model/RAC_classifier.pkl')

# prediction process
Pred_result = []
for batch_idx,(lines,properties) in enumerate(val_loader, 1):
    input, seq_lengths, y = make_variables(lines, properties)
    p_model.batch_size = input.size(0)
    pred,wts = p_model(input)
    pred = torch.max(pred.data.cpu(),1)[1]
    Pred_result.append(pred.cpu().data.numpy())
Pred_results = [j for i in  Pred_result for j in i.tolist()]
pattern = {0:'2N', 1:'2Y', 2:'3N', 3:'3Y'}
P = [pattern[x] if x in pattern else x for x in Pred_results]
# output results
results = pd.DataFrame(P)
results.to_csv("./Prediction_Result/results.csv", index=None)

