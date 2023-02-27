# performance function to eval correlations between true and pred
import numpy as np
import os
from numpy.core.numeric import rollaxis
from sklearn.metrics import matthews_corrcoef,accuracy_score,precision_score,recall_score,roc_auc_score, confusion_matrix,balanced_accuracy_score,roc_curve,precision_recall_curve,f1_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, average="binary")
def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, average="binary")
def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)
def new_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
def sp(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
def se(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[1,1] * 1.0 / (cm[1,1] + cm[1,0])
def bacc(y_true, y_pred):
    return balanced_accuracy_score(y_true,y_pred)
def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)
def auc_curve(y_true,y_score):
    fpr,tpr,threshold = roc_curve(y_true,y_score)
    return fpr,tpr,threshold

def performance_function(y_true,y_pred_proba):
    y_pred = y_pred_proba
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    return [auc(y_true,y_pred_proba),accuracy(y_true,y_pred),bacc(y_true,y_pred),recall(y_true,y_pred),precision(y_true, y_pred),se(y_true,y_pred),sp(y_true,y_pred)]

def AllMetrics(y_true,y_score,Assay = os.getpid()):
    y_pred = np.copy(y_score)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred <  0.5] = 0
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    StorageDict = {x:[] for x in ['Assay','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC','y_true','y_score']}
    StorageDict['Assay'].append(Assay)
    StorageDict['AUC'].append(roc_auc_score(y_true,y_score))
    StorageDict['AUPR'].append(auc(recalls,precisions))
    StorageDict['Accuracy'].append(accuracy_score(y_true,y_pred))
    StorageDict['BalancedAccuracy'].append(balanced_accuracy_score(y_true,y_pred))
    StorageDict['Precision'].append(precision_score(y_true, y_pred, pos_label=1, average="binary"))
    StorageDict['Recall'].append(recall_score(y_true, y_pred, pos_label=1, average="binary"))
    StorageDict['F1Score'].append(f1_score(y_true,y_pred, pos_label=1, average='binary'))
    StorageDict['Sensitivity'].append(float(tp) / (tp + fn))
    StorageDict['Specificity'].append(float(tn) / (tn + fp))
    StorageDict['MCC'].append((tp * tn - fp * fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)))
    StorageDict['y_true'].append(y_true)
    StorageDict['y_score'].append(y_score)
    return StorageDict