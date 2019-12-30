# coding: utf-8

"""
    Scripts for training BiLSTM-SA model.

"""
import sys
sys.path.append('Utils/')
import Smipar

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

import argparse
import log as lg
import warnings
warnings.filterwarnings("ignore")

# batch_size=64, lstm_hid_dim=128, d_a = 100, r=10,
def parse_args():
    """Parses arguments from terminal"""
    
    parser = argparse.ArgumentParser(description="Generate products from a batch of given reactants' SMILES.")
    
    parser.add_argument("--lstm_hid_dim", "-lhd",
                        help=("how many hidden units in LSTM layer."), type=int, required=True)
    
    parser.add_argument("--d_a", "-da", 
                        help="self-attention-para.", type=int, required=True)
    
    parser.add_argument("--r", "-r", 
                        help="self-attention-para.", type=int, required=True)
    
    parser.add_argument("--epochs", "-e", 
                        help="model-para.", type=int, required=True)
    
    parser.add_argument("--output_model_path", "-o", 
                        help="Prefix to the output model.", type=str, required=True)
    
    
    return parser.parse_args()


class ProDataset(Dataset):
    """ load dataset."""
    
    def __init__(self, is_train_set=False):
        if is_train_set:
            df = pd.read_csv("Dataset/train.csv")
        else:
            df = pd.read_csv("Dataset/test.csv")
        
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
  
batch_size = 2
train_dataset = ProDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = ProDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    
with open('Dataset/Voc', 'r') as f: 
    tokens = f.read().split()
all_letters = tokens
N_CHARS = len(all_letters)

def create_variable(tensor):
    "utility functions"
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
    
def tokenToIndex(token):
    "used to get index as voc order"
    return all_letters.index(token)


def reaction_str2ascii_arr(string):
    "used to put tokens into acsii index,ord() will return the index of ascii ie. ord(c) =99"
    arr = [ord(c) for c in string]
    return arr, len(arr)


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


# simply modify the code for this paper A Structured Self-Attentive Sentence Embedding
class StructuredSelfAttention(torch.nn.Module):

    def __init__(self, batch_size, lstm_hid_dim, d_a, r, max_len, emb_dim=128, vocab_size=None,
                 use_pretrained_embeddings = False, embeddings=None, type=1, n_classes = 4, bidirectional=True):
        """
        Initializes parameters 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
        Returns:
            self
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
        self.emb_dim = emb_dim 
        self.embeddings= nn.Embedding(vocab_size, emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim, lstm_hid_dim, 1, batch_first=True, bidirectional=True) 
        if  bidirectional:
            self.bi_num=2
        else:
            self.bi_num=1
        self.linear_first = torch.nn.Linear(self.bi_num*lstm_hid_dim, d_a) 
        self.linear_first.bias.data.fill_(0) 
        self.linear_second = torch.nn.Linear(d_a, r) 
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(self.bi_num*lstm_hid_dim, self.n_classes) 
        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.r = r
        self.type = type
        
    def softmax(self, input, axis=1):

        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
    def init_hidden(self):
        
        return (Variable(torch.zeros(2,self.batch_size,self.lstm_hid_dim).cuda()),
                Variable(torch.zeros(2,self.batch_size,self.lstm_hid_dim).cuda())) 
        
    def forward(self, x):
        embeddings = self.embeddings(x)       
        outputs, self.hidden_state = self.lstm(embeddings, self.hidden_state) 
        x = F.tanh(self.linear_first(outputs))    
        x = self.linear_second(x)     
        x = self.softmax(x, 1) 
        attention = x.transpose(1, 2)
        sentence_embeddings = attention@outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
               
        if not bool(self.type):
            output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
            return output, attention
        else:
            return F.log_softmax(self.linear_final(avg_sentence_embeddings)), attention # run this 
        
    #Regularization
    def l2_matrix_norm(self, m):
        """
        Frobenius norm calculation
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor).cuda()
    
    
    
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
            
            if batch_idx % 50==0: 
                print(batch_idx)
            

        yy_target = [j for i in target_train for j in i]
        yy_pred = [j for i in pred_train for j in i]       
        print("Train_accuracy_sklearn",accuracy_score(yy_target, yy_pred))
        train_binarize = label_binarize(yy_target, classes=[0, 1, 2, 3])
        y_train_proba = output_proba[1:]
        print("Training AUC:", roc_auc_score(train_binarize, y_train_proba))

        target_names = ['2N', '2Y', '3N', "3Y"]
        print("Train classification_report:", classification_report(yy_target, yy_pred, target_names=target_names))

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


def main():
    
    args = parse_args()

    LOG.info("Define model")
    attention_model = StructuredSelfAttention(batch_size=batch_size, lstm_hid_dim=args.lstm_hid_dim, \
                                              d_a=args.d_a, r=args.r, bidirectional=True, 
                                          vocab_size=N_CHARS, type=1, n_classes=4, max_len=150) 
    #attention_model.cuda()

    multi_classfication(attention_model, train_loader=train_loader, epochs=args.epochs, use_regularization=False,C=0.3,clip=True)

    torch.save(attention_model, args.output_model_path)
    
    
LOG = lg.get_logger(name="training model and save model")

if __name__ == "__main__":
    main()
