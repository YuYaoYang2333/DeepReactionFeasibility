# coding: utf-8
"*****************************************************"
# Code for classification by SVM OneVsRestClassifier
# Descriptors : RFC_QM_DES
# split method : Random
"*****************************************************"
# The strategy consists in fitting one classifier per class. one advantage of this approach is its interpretability.

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler， label_binarize
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score， cohen_kappa_score，confusion_matrix，hamming_loss，precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

def load_data(filename):
    data = pd.read_csv(filename)
    data.drop(data.columns[[0,1,2]], axis=1,inplace=True)
    return data

df = load_data("RFC_QM_DES.csv") # load data

multi_model = OneVsRestClassifier(SVC(probability=True)) # best result

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
cv_test_pred = []
Y_label_test = []
cv_train_pred = []
Y_label_train = []
prob_train = np.zeros((1, 4)) # 4 labels
prob_test = np.zeros((1, 4))

for i, (train_idx, test_idx) in enumerate(kf.split(df)):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    #print(len(train_idx))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    train = train.sample(frac = 1) 
    test = test.sample(frac = 1) 
    y_train = train["type_2"]
    #y_train = label_binarize(y_train, classes=[0, 1, 2, 3]) 
    #y_train = np.array(y_train).reshape(y_train.shape[0], 4) 
    y_test = test["type_2"]
    #y_test = label_binarize(y_test, classes=[0, 1, 2, 3])
    #y_test = np.array(y_test).reshape(y_test.shape[0], 4)
    model_x_train = train.copy()
    model_x_test =   test.copy()
    model_x_train.drop(model_x_train.columns[[0]], axis=1,inplace=True)
    x_train = np.array(model_x_train)
    model_x_test.drop(model_x_test.columns[[0]], axis=1,inplace=True)
    x_test = np.array(model_x_test)
   # processing data
    x_data = np.vstack((x_train, x_test))
    scaler = StandardScaler().fit(x_data)
    #print(scaler.mean_ )
    #print(scaler.std_ )
    x_train =  scaler.transform(x_train)
    x_test =  scaler.transform(x_test)
    # model trainning
    multi_model.fit(x_train, y_train)
    
    train_pred = multi_model.predict(x_train)
    test_pred = multi_model.predict(x_test)
    cv_test_pred.append(test_pred)
    Y_label_test.append(y_test)
    cv_train_pred.append(train_pred)
    Y_label_train.append(y_train)
    #prob_train = np.vstack((prob_train, x_train))
    #prob_test = np.vstack((prob_test, x_test))
    #train_accuracy = accuracy_score(y_train, train_pred)
    print("Train_accuracy",accuracy_score(y_train, train_pred))
    train_binarize = label_binarize(y_train, classes=[0, 1, 2, 3])
    y_train_proba = multi_model.predict_proba(x_train)
    prob_train = np.vstack((prob_train, y_train_proba))
    print("Train AUC:", roc_auc_score(train_binarize, y_train_proba, average = 'macro')) #..ravel()# macro micro # macro 
    print("Train cohen_kappa_score:", cohen_kappa_score(y_train, train_pred))
    print("Train hamming_loss:", hamming_loss(y_train, train_pred))
    target_names = ['2N', '2Y', '3N', "3Y"]
    print("Train classification_report:", classification_report(y_train, train_pred, target_names=target_names))
    print ("Train precision_score_macro:", precision_score(y_train, train_pred,average='macro')) # precision
    print("Train recall_score_macro:", recall_score(y_train, train_pred,average='macro')) # recall
    print("Train confusion_matrix:", confusion_matrix(y_train, train_pred))
    print("--------------------------------------------------------------------------------------")
    print("Test_accuracy", accuracy_score(y_test, test_pred))
    test_binarize = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_test_proba = multi_model.predict_proba(x_test)
    prob_test = np.vstack((prob_test, y_test_proba))
    print("Test AUC:", roc_auc_score(test_binarize, y_test_proba, average = 'macro'))
    print("Test cohen_kappa_score:", cohen_kappa_score(y_test, test_pred))
    print("Test hamming_loss:", hamming_loss(y_test, test_pred))
    target_names = ['2N', '2Y', '3N', "3Y"]
    print("Test classification_report:", classification_report(y_test, test_pred, target_names=target_names))
    print ("Test precision_score_macro:", precision_score(y_test, test_pred,average='macro')) # precision
    print("Test recall_score_macro:", recall_score(y_test, test_pred,average='macro')) # recall
    #print(accuracy_score(y_test, test_pred, normalize=False)) # correct
    print("Test confusion_matrix:", confusion_matrix(y_test, test_pred))
    print("*****************************************************************")
    
print("****************************************************************")
cv_1_test =  [m for i in  cv_test_pred for m in i.tolist()]
Y_1=  [m for i in Y_label_test for m in i.tolist()]
cv_1_test = np.array(cv_1_test)
Y_1 = np.array(Y_1)

print("Test_accuracy", accuracy_score(Y_1, cv_1_test))
test_binarize = label_binarize(Y_1, classes=[0, 1, 2, 3])
prob_test = prob_test[1:]
print("Test AUC:", roc_auc_score(test_binarize, prob_test, average = 'macro'))
print("Test cohen_kappa_score:", cohen_kappa_score(Y_1, cv_1_test))
print("Test hamming_loss:", hamming_loss(Y_1, cv_1_test))
target_names = ['2N', '2Y', '3N', "3Y"]
print("Test classification_report:", classification_report(Y_1, cv_1_test, target_names=target_names))
print ("Test precision_score_macro:", precision_score(Y_1, cv_1_test,average='macro')) # precision
print("Test recall_score_macro:", recall_score(Y_1, cv_1_test,average='macro')) # recall
#print(accuracy_score(y_test, test_pred, normalize=False)) # correct
print("Test confusion_matrix:", confusion_matrix(Y_1, cv_1_test))

print("****************************************************************")
cv_2_train =  [m for i in  cv_train_pred for m in i.tolist()]
Y_2=  [m for i in Y_label_train for m in i.tolist()]
cv_2_train = np.array(cv_2_train)
Y_actural = np.array(Y_2)
print("Train_accuracy",accuracy_score(Y_2, cv_2_train))
train_binarize = label_binarize(Y_2, classes=[0, 1, 2, 3])
prob_train = prob_train[1:]
print("Train AUC:", roc_auc_score(train_binarize, prob_train, average = 'macro')) #..ravel()# macro micro # macro 
print("Train hamming_loss:", hamming_loss(Y_2, cv_2_train))
target_names = ['2N', '2Y', '3N', "3Y"]
print("Train classification_report:", classification_report(Y_2, cv_2_train, target_names=target_names))
print ("Train precision_score_macro:", precision_score(Y_2, cv_2_train,average='macro')) # precision
print("Train recall_score_macro:", recall_score(Y_2, cv_2_train,average='macro')) # recall
print("Train confusion_matrix:", confusion_matrix(Y_2, cv_2_train))
