#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


# here put the import lib
import os,time
from numpy.core.fromnumeric import product
import pandas as pd
import numpy as np

from itertools import product
import deepchem as dc
from deepchem.models import GCNModel,GATModel,AttentiveFPModel
from deepchem.models.torch_models import MPNNModel
from pandas.core import algorithms
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from SamplingStrategies import Resampling
from performance_evaluation import AllMetrics


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

global RANDOMSATAE
RANDOMSTATE = 30191375
np.random.seed(RANDOMSTATE)
tf.random.set_seed(RANDOMSTATE)
RootDir = os.getcwd()
DataDir = os.path.join(RootDir,'data')
target = 'CYP3A4'
kfold = 10
EPOCH = 10
FEATURIZERS = ['GraphConv']
#ALGORITHMS = ['GCN','GNN']
ALGORITHMS = ['GCN']
SAMPLING = ['Original','ROS','RUS']

RootDir = os.getcwd()
DataDir = os.path.join(RootDir,'data')
name = 'CYP3A4'
try:
    os.mkdir('local')        
    os.mkdir('local/models')
    os.mkdir('local/temp')
except Exception:
    pass

PARAMETERS = {
    'GCN' : {
        "learning_rate" : [0.005,0.01,0.1],
        "activation" : [None],
        "batchnorm" : [True,False],
        "dropout" : [0,0.1,0.25,0.5],
        "uncertainty" : [True],
        "output_types" : ['loss']
    },
}
kf = StratifiedKFold(n_splits = 10,shuffle=True,random_state = 100)
def model_builder(model,model_params,model_dir):
    return model(**model_params,model_dir = model_dir)
def CheckHeader(filepath):
    if os.path.exists(filepath):
        AddHeader = False
    else:
        AddHeader = True
    return AddHeader
def model_bulider(**model_params):
    batch_size = model_params['batch_size']
    n_tasks = model_params['n_tasks']
    mode = model_params['mode']
    batch_size = model_params["batch_size" ]
    learning_rate = model_params["learning_rate" ]
    batchnorm = model_params["batchnorm" ]
    dropout = model_params["dropout" ]
    output_types = model_params["output_types" ]
    return dc.models.GraphConvModel(n_tasks=1, mode='classification', batch_size=64,learning_rate=learning_rate,batchnorm =batchnorm ,dropout =dropout,output_types = output_types)
def build_deepchem_model(train_dataset,valid_dataset,algorithm_name,model_dir):
    algorithm_map = {'GCN': GCNModel,'GAT': GATModel,'AttentiveFP' : AttentiveFPModel,'MPNN' : MPNNModel}
    tuned_parameters = PARAMETERS[algorithm_name]
    print(tuned_parameters)
    method = algorithm_map[algorithm_name]
    grid = dc.hyper.GridHyperparamOpt(model_builder)
    '''
    for train_index,test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    '''  
    best_model, best_hyperparams, all_results = grid.hyperparam_search(tuned_parameters,train_dataset,valid_dataset,[dc.metrics.Metric
    (dc.metrics.roc_auc_score),dc.metrics.Metric
    (dc.metrics.accuracy_score),dc.metrics.Metric
    (dc.metrics.balanced_accuracy_score),dc.metrics.Metric
    (dc.metrics.f1_score),dc.metrics.Metric
    (dc.metrics.matthews_corrcoef)],logdir = model_dir)
    return best_model,best_hyperparams,all_results

original = pd.read_csv(os.path.join(DataDir,'NewIdea.csv')).rename(columns={'CanonicalSMILES':'Canonical_Smiles'})
data = original
unique = data.drop_duplicates('Canonical_Smiles').reset_index(drop = True)
#prepared = unique.loc[unique.outlier.isna()]
label = unique['CYP3A4']
print(name,len(label),Counter(label))
cansmi = unique['Canonical_Smiles']

for featurizer,algorithm,sampling in product(FEATURIZERS,ALGORITHMS,SAMPLING):
    name_str = f'{name}_{featurizer}_{algorithm}_{sampling}'
    trainset = pd.read_csv(os.path.join(DataDir,'sampling',f'Train-{sampling}.csv'))
    testset = pd.read_csv(os.path.join(DataDir,'sampling',f'Test-{sampling}.csv'))
    validset = pd.read_csv(os.path.join(DataDir,'sampling',f'Valid-{sampling}.csv'))
    x_train = trainset['Canonical_Smiles']
    x_test = testset['Canonical_Smiles']
    x_valid = validset['Canonical_Smiles']
    y_train = trainset['CYP3A4']
    y_test = testset['CYP3A4']
    y_valid  = validset['CYP3A4']


    loader = dc.data.InMemoryLoader(tasks = [target],featurizer=dc.feat.ConvMolFeaturizer())
    train_set = loader.create_dataset(zip(x_train,y_train),shard_size = 2)
    test_set = loader.create_dataset(zip(x_test,y_test),shard_size = 2)
    valid_set = loader.create_dataset(zip(x_valid,y_valid),shard_size = 2)
    model = dc.models.GraphConvModel(n_tasks=1, mode='classification',dropout = 0.1,model_dir = f'local/models/{name_str}.m')
    model.fit(train_set,nb_epoch = EPOCH)
    #pd.DataFrame(cv_res).to_csv(f'local/temp/{name_str}_cv.csv',index = False)
    out = model.predict(train_set)[:,:,1].ravel()
    

    AddHeader = CheckHeader(f'local/temp/{name}_train.csv')
    result = [name,featurizer,algorithm,sampling,str({'dropout':0.1})]+list(np.array(list(AllMetrics(y_train,out).values())).ravel())
    pd.DataFrame([result],columns = ['endpoint','fp','method','sampling','param','Assay','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC','y_true','y_score']).to_csv(f'local/temp/{name}_train.csv',index = False,mode = 'a',header= AddHeader)

    # test
    
    proba = model.predict(test_set)[:,:,1].ravel()
    print(proba)
    r = [name,featurizer,algorithm,sampling]+list(np.array(list(AllMetrics(y_test,proba).values())).ravel())
    pd.DataFrame([r],columns =['endpoint','fp','method','sampling','Assay','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC','y_true','y_score']).to_csv(f'local/temp/{name}_test.csv',index = False,mode = 'a',header= AddHeader)
    
    validproba = model.predict(valid_set)[:,:,1].ravel()
    r = [name,featurizer,algorithm,sampling]+list(np.array(list(AllMetrics(y_valid,validproba).values())).ravel())
    pd.DataFrame([r],columns =['endpoint','fp','method','sampling','Assay','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC','y_true','y_score']).to_csv(f'local/temp/{name}_valid.csv',index = False,mode = 'a',header= AddHeader)
    

'''
    splitter = splits.RandomStratifiedSplitter()
    cv_sets = splitter.k_fold_split(train_set,kfold)
    cv_result = pd.DataFrame()
    for i in range(kfold):
        cv_set = cv_sets[i][1]
        print(f'[cv{i}]',pd.DataFrame(cv_set.y)[0].value_counts())
        index = list(range(kfold))
        index.remove(i)
        cv_train =  dc.data.DiskDataset.merge([cv_sets[j][1] for j in index])
        cv_test = cv_set
        print(cv_test)
        model = dc.models.GraphConvModel(n_tasks=1, mode='classification', batch_size=64,learning_rate=0.005,model_dir = os.path.join(RootDir,'local','models',f'CYP3A4_GraphConv_GCN_cv{i}.m'))
        model.fit(cv_train,nb_epoch = EPOCH)
        outcome = metrics(model,cv_test,'cv')
        print(f'cv_{i}:{outcome}')
        cv_result = pd.concat([cv_result,outcome])

    pd.DataFrame(cv_result.mean(),columns=['cv']).T.reset_index(drop=False).rename(columns={'index':'model'}).to_csv(os.path.join(RootDir,'local','temp','GCN.csv'),index=False)

    # test
    print('retrain on whole train set')
    model = dc.models.GraphConvModel(n_tasks=1, mode='classification', batch_size=64,learning_rate=0.005,model_dir = os.path.join(RootDir,'local','models','CYP3A4_GraphConv_GCN.m'))
    model.fit(train_set,nb_epoch=EPOCH)
    test_out = metrics(model,test_set,'test')
    valid_out = metrics(model,valid_set,'valid')
    test_out.to_csv(os.path.join(RootDir,'local','temp','GCN.csv'),mode='a',index=False,header=False)
    valid_out.to_csv(os.path.join(RootDir,'local','temp','GCN.csv'),mode='a',index =False,header=False)

'''

'''
############################
# AttentiveFP model

AttentiveFPloader = dc.data.CSVLoader([target],feature_field = 'Canonical_Smiles',featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True))
train_set = AttentiveFPloader.create_dataset(os.path.join(DataDir,'train.csv'))
test_set = AttentiveFPloader.create_dataset(os.path.join(DataDir,'test.csv'))
valid_set = AttentiveFPloader.create_dataset(os.path.join(DataDir,'valid.csv'))
print('train',pd.DataFrame(train_set.y)[0].value_counts())
print('test',pd.DataFrame(test_set.y)[0].value_counts())
print('valid',pd.DataFrame(valid_set.y)[0].value_counts())

splitter = splits.RandomStratifiedSplitter()
cv_sets = splitter.k_fold_split(train_set,kfold)
cv_result = pd.DataFrame()
for i in range(kfold):
    cv_set = cv_sets[i][1]
    print(f'[cv{i}]',pd.DataFrame(cv_set.y)[0].value_counts())
    index = list(range(kfold))
    index.remove(i)
    cv_train =  dc.data.DiskDataset.merge([cv_sets[j][1] for j in index])
    cv_test = cv_set
    print(cv_test)
    model = dc.models.AttentiveFPModel(n_tasks=1, mode='classification', batch_size=64,learning_rate=0.005,model_dir = os.path.join(RootDir,'local','models',f'CYP3A4_AttentiveFP_GNN_cv{i}.m'))
    model.fit(cv_train,nb_epoch = EPOCH)
    outcome = metrics(model,cv_test,'cv')
    print(f'cv_{i}:{outcome}')
    cv_result = pd.concat([cv_result,outcome])

pd.DataFrame(cv_result.mean(),columns=['cv']).T.reset_index(drop=False).rename(columns={'index':'model'}).to_csv(os.path.join(RootDir,'local','temp','GNN.csv'),index=False)

# test
print('retrain on whole train set')
model = dc.models.AttentiveFPModel(n_tasks=1, mode='classification', batch_size=64,learning_rate=0.005,model_dir = os.path.join(RootDir,'local','models','CYP3A4_AttentiveFP_GNN.m'))
model.fit(train_set,nb_epoch=EPOCH)
test_out = metrics(model,test_set,'test')
valid_out = metrics(model,valid_set,'valid')
test_out.to_csv(os.path.join(RootDir,'local','temp','GNN.csv'),mode='a',index=False,header=False)
valid_out.to_csv(os.path.join(RootDir,'local','temp','GNN.csv'),mode='a',index =False,header=False)
'''