

# coding: utf-8


import re
import string
import Smipar
from model import *
import torch, keras
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

#data processing
class ProDataset(Dataset):
    """ load dataset."""
    # Initialize your data, download, etc.
    def __init__(self, is_train_set=False):
        df = Train.copy() if is_train_set else Test.copy()
        
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
		
# Voc loaders
with open('./Dataset/Voc-RAC', 'r') as f: 
    tokens = f.read().split()
all_letters = tokens
N_CHARS = len(all_letters)

def get_smiles_a(string):
    split_rs = string.split('>')
    product = split_rs[1]
    reactants = split_rs[0].split('.')
    reactant_a = reactants[0]
    return reactant_a

def create_variable(tensor):
    "utility functions"
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
    
def  judge_sp2_sp3(smiles):
    "get a a-smiles , judge if the C connecting with the Br is sp2/sp3"
    all_sp2 = Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmarts('[^2]'))
    C_number = Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmiles("CBr"))
    all_sp2_list = [j for i in all_sp2 for j in i]
    C_number_list = [j for i in C_number for j in i]
    common = [i for i in C_number_list if i in all_sp2_list ]
    length = len(common)
    if length == 0:
        type_sp = "sp3"
    else:
        type_sp = "sp2"
    return type_sp
     
def tokenToIndex(token):
    "used to get index as voc order"
    return all_letters.index(token)
# print(tokenToIndex('[nH]'))

def reaction_str2ascii_arr(string):
    "used to put tokens into acsii index,ord() will return the index of ascii ie. ord(c) =99"
    arr = [ord(c) for c in string]
    return arr, len(arr)
# print(reaction_str2ascii_arr('cCC([nH]'))

def reaction_str2voc_arr(string):
    "used to put reaction string into voc index"
    arr = []
    "Spilt the reaction string into two reactants and a products"
    split_rs = string.split('>')
    product = split_rs[1]
    reactants = split_rs[0].split('.')
    reactant_a = reactants[0]
    reactant_e = reactants[1]
    debris = Smipar.parser_list(reactant_a) + ['.'] + Smipar.parser_list(reactant_e) +['>'] + Smipar.parser_list(product)
    for i, d in enumerate(debris):
        arr.append(tokenToIndex(d))
    return arr, len(arr)

def properties2tensor(properties):
    property_ids = [train_dataset.get_property_id(property) for property in properties]
    return torch.LongTensor(property_ids)


def pad_sequences(vectorized_seqs, seq_lengths, properties):
    "pad sequences and sort the tensor"
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target in the same order
    target = properties2tensor(properties)
    if len(properties):
        target = target[perm_idx]

    return create_variable(seq_tensor),         create_variable(seq_lengths),         create_variable(target)

def make_variables(lines, properties): 
    "Create necessary variables, lengths, and target"
    sequence_and_length = [reaction_str2voc_arr(line) for line in lines] 
    vectorized_seqs = [sl[0] for sl in sequence_and_length]# arr
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length]) # len(arr)
    return pad_sequences(vectorized_seqs, seq_lengths, properties)

# Parameters and DataLoaders
BATCH_SIZE = 1
N_EPOCHS = 20

def train(attention_model,train_loader,criterion,optimizer,epochs = 20, use_regularization = False,C=0, clip=False):
    """
        Training code
        Args:
            attention_model : {object} model
            train_loader    : {DataLoader} training data loaded into a dataloader
            optimizer       :  optimizer
            criterion       :  loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
            epochs          : {int} number of epochs
            use_regularizer : {bool} use penalization or not
            C               : {int} penalization coeff
            clip            : {bool} use gradient clipping or not 
        Returns:
            accuracy and losses of the model
        """
    losses = []
    accuracy = []
    roc = []
    for i in range(epochs):
        print("Running EPOCH",i+1)
        total_loss = 0 
        n_batches = 0
        correct = 0
        output_proba = np.zeros((1, 4))
        pred_train = []
        target_train = []
        
        for batch_idx,(lines, properties) in enumerate(train_loader, 1):
            input, seq_lengths, y = make_variables(lines, properties)
            attention_model.batch_size = y.shape[0]
            attention_model.hidden_state = attention_model.init_hidden()
            y_pred, att = attention_model(input)
            # add judge sp2/sp3
            if use_regularization:
                attT = att.transpose(1,2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size, att.size(1),att.size(1))).cuda()
                penal = attention_model.l2_matrix_norm(att@attT - identity)
                
            # nulti-classification
            correct+=torch.eq(torch.max(y_pred.data.cpu(),1)[1],y.type(torch.LongTensor)).data.sum()
            
            if use_regularization:
                loss = criterion(y_pred,y) + (C * penal/train_loader.batch_size).type(torch.FloatTensor)
            else:
                loss = criterion(y_pred,y)
                
            output_proba = np.vstack((output_proba, y_pred.data.cpu().numpy()))
            pred = torch.max(y_pred.data.cpu(),1)[1]
            pred_train.append(pred.data.cpu().numpy())
            target_train.append(y.data.cpu().numpy())
            
            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward()
            
            #gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(),0.5)
            optimizer.step()
            n_batches+=1
            
            if batch_idx % 500==0: 
                print(batch_idx)
            

        yy_target = [j for i in target_train for j in i]
        yy_pred = [j for i in pred_train for j in i]       
        print("Train_accuracy_sklearn",accuracy_score(yy_target, yy_pred))
        train_binarize = label_binarize(yy_target, classes=[0, 1, 2, 3])
        y_train_proba = output_proba[1:]
        print("Training AUC:", roc_auc_score(train_binarize, y_train_proba))
        #print("Train cohen_kappa_score:", cohen_kappa_score(y_train, train_pred))
        target_names = ['2N', '2Y', '3N', "3Y"]
        print("Train classification_report:", classification_report(yy_target, yy_pred, target_names=target_names))
        #print ("Train precision_score_macro:", precision_score(y_train, train_pred,average='macro')) # precision
        #print("Train recall_score_macro:", recall_score(y_train, train_pred,average='macro')) # recall
        print("Train confusion_matrix:", confusion_matrix(yy_target, yy_pred))
        # model 
        print("avg_loss is",total_loss/n_batches)
        print("Accuracy of the model",correct.numpy()/(len(train_loader.dataset)))
        test_loss,test_acc = test(attention_model,test_loader,loss)
        losses.append(test_loss)
        accuracy.append(test_acc)
    return losses, accuracy

# torch.nn.NLLLoss()
def test(attention_model,test_loader,criterion):
    """
        Training code for test
        Args:
            attention_model : {object} model
            train_loader    : {DataLoader} training data loaded into a dataloader
            optimizer       :  optimizer
            criterion       :  loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
            epochs          : {int} number of epochs
            use_regularizer : {bool} use penalization or not
            C               : {int} penalization coeff
            clip            : {bool} use gradient clipping or not
        Returns:
            accuracy and losses of the model
        """
    losses = []
    accuracy = []
    print("Test ...")
    for i in range(1):
        total_loss = 0
        n_batches = 0
        correct = 0
        recall = 0
        AUC = 0
        all_pred = np.array([])
        all_target = np.array([])
        totalpred = 0
        output_proba = np.zeros((1, 4))
        pred_test = []
        target_test = []
        for batch_idx,(lines,properties) in enumerate(test_loader, 1):  
            
            input, seq_lengths, y = make_variables(lines, properties)
            attention_model.batch_size = y.shape[0] 
            attention_model.hidden_state = attention_model.init_hidden()
            y_pred, att = attention_model(input)

            pred = torch.max(y_pred.data.cpu(),1)[1]
            correct+=torch.eq(torch.max(y_pred.data.cpu(),1)[1],y.type(torch.LongTensor)).data.sum()
            output_proba = np.vstack((output_proba, y_pred.data.cpu().numpy()))
            pred_test.append(pred.data.cpu().numpy())
            target_test.append(y.data.cpu().numpy())
                
            n_batches+=1
        print(len(test_loader.dataset)) 
        
        yy_target = [j for i in target_test for j in i]
        yy_pred = [j for i in pred_test for j in i]       
        print("Test_accuracy_sklearn",accuracy_score(yy_target, yy_pred))
        test_binarize = label_binarize(yy_target, classes=[0, 1, 2, 3])
        y_test_proba = output_proba[1:]
        print("Test AUC:", roc_auc_score(test_binarize, y_test_proba))
        target_names = ['2N', '2Y', '3N', "3Y"]
        print("Test classification_report:", classification_report(yy_target, yy_pred, target_names=target_names))
        print("Test confusion_matrix:", confusion_matrix(yy_target, yy_pred))
        # model
        print("Accuracy of the test set",correct.numpy()/(n_batches*test_loader.batch_size))
        #print("recall =",metrics.recall_score(np.round(all_pred),all_target))
        #print("recall_own =", recall/n_batches)
        #print("size = ",len(all_pred),len(all_target))
        #print("AUC = ",metrics.roc_auc_score(all_target, np.round(all_pred)))
        print("test loss = {}".format(total_loss/n_batches))
        acc_own = accuracy_score(yy_target, yy_pred)
        test_acc = (correct.numpy()/(n_batches*test_loader.batch_size))
    return acc_own, test_acc

def multi_classfication(attention_model, train_loader,epochs=20, use_regularization=True, C=1.0,clip=True):
    attention_model.cuda()
    loss = torch.nn.NLLLoss() 
    optimizer = torch.optim.Adam(attention_model.parameters())
    losses, accuracy = train(attention_model,train_loader,loss,optimizer,epochs,use_regularization,C,clip)
    return losses, accuracy

# load data
data= pd.read_csv(r"./Dataset/your-data.csv") # your own training dataset filename
data =  data.sample(frac = 1)
kf = KFold(n_splits=5, random_state=1234, shuffle=True)
for i, (train_idx, test_idx) in enumerate(kf.split(data)):
    Train = data.iloc[train_idx]
    Test  = data.iloc[test_idx]
    
train_dataset = ProDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = ProDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

import warnings
warnings.filterwarnings("ignore")

attention_model = StructuredSelfAttention(batch_size=BATCH_SIZE, lstm_hid_dim=128, d_a = 100, r=10, bidirectional=True, 
                                          vocab_size=N_CHARS, type=1, n_classes=4, max_len=150) 
#attention_model.cuda()

multi_classfication(attention_model, train_loader=train_loader, epochs=20, use_regularization=False,C=0.3,clip=True)

torch.save(attention_model, r'./Model/your_own_model.pkl')

