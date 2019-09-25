

# coding: utf-8

# Simply modify the code for this paper A Structured Self-Attentive Sentence Embedding
# (published in ICLR 2017: https://arxiv.org/abs/1703.03130)


import Smipar
import torch, keras
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix
### define attention layer
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