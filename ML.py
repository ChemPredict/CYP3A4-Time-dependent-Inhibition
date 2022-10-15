from operator import mod
import os 
import pandas as pd 
import numpy as np 
from itertools import product
import math

from pandas.core.frame import DataFrame
import time 
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import Counter
from performance_evaluation import accuracy,recall,precision,AUC,bacc,aupr,f1,sensitivity,specificity,MCC,AllMetrics
#from notification import JobDone
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer,plot_roc_curve,plot_precision_recall_curve,precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from SamplingStrategies import Resampling
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    
global PADELFPS,METHODS,RANDOMSATAE,N_JOBS
RANDOMSTATE = 30191375
PADELFPS = ['MACCS', 'CDKExt', 'CDK', 'EState', 'GraphFP', 'PubChemFP', 'SubFP', 'KRFP', 'AP2D']
METHODS = ['svm', 'nn', 'rf', 'knn']
#SAMPLING = ['Original','ROS','SMOTE','RUS','CC']
SAMPLING = ['Original','ROS','RUS']
#PADELFPS = ['KRFP']
#METHODS = ['svm']
np.random.seed(RANDOMSTATE)
N_JOBS = math.floor(os.cpu_count()*100/100)
#N_JOBS = -1
#N_JOBS = 18
'''
PARAMETERS = {
    'svm': {'kernel': ['rbf'], 'gamma': 2**np.arange(-3,-15,-2, dtype=float),
            'C':2**np.arange(-5,15,2, dtype=float),
            'class_weight':['balanced'], 'cache_size':[400] },
    'nn': {"learning_rate":["adaptive"],   #learning_rate:{"constant","invscaling","adaptive"}默认是constant
            "max_iter":[10000],
            "hidden_layer_sizes":[(100,),(500,),(1000,),(200,100),(200,100,50),(100,200,100)],
            "alpha":10.0 ** -np.arange(1, 7),
            "activation":["relu"],  #"identity","tanh","relu","logistic"
            "solver":["adam"],     #"lbfgs" for small dataset
            'warm_start':[True]},
    'knn': {"n_neighbors":range(2,15,1),"weights":['distance'],'p':[1,2],
            'metric':['minkowski','jaccard']},
    'rf': {"n_estimators":range(10,501,20),
            "criterion" : ["gini"], #['entropy']
            "oob_score": ["False"],
            "class_weight":["balanced_subsample"]}
}
'''
PARAMETERS = {
    'svm': {'kernel': ['rbf'], 'gamma': 2**np.arange(-3,-15,-2, dtype=float),
            'C':2**np.arange(-5,15,2, dtype=float),
            'cache_size':[400] },
    'nn': {"learning_rate":["adaptive"],   #learning_rate:{"constant","invscaling","adaptive"}默认是constant
            "max_iter":[5000],
            "hidden_layer_sizes":[(100,),(500,),(1000,),(200,100),(200,100,50),(100,200,100)],
            "alpha":10.0 ** -np.arange(1, 7),
            "activation":["relu"],  #"identity","tanh","relu","logistic"
            "solver":["adam"],     #"lbfgs" for small dataset
            'warm_start':[True]},
    'knn': {"n_neighbors":range(2,15,1),'p':[1,2],
            'metric':['minkowski','jaccard']},
    'rf': {"n_estimators":range(10,501,20),
            "criterion" : ["gini"], #['entropy']
            "oob_score": ["False"],
            }
}
#SCORING_FNC = {'SE':make_scorer(se),'SP':make_scorer(sp),'AUC':make_scorer(auc),'ACC':'accuracy','bACC':make_scorer(bacc)}
SCORING_FNC = {'AUC':make_scorer(AUC),'AUPR':make_scorer(aupr),'Accuracy':'accuracy','BalancedAccuracy':make_scorer(bacc),'Precision':'precision','Recall':make_scorer(recall),'F1Score':make_scorer(f1),'Sensitivity':make_scorer(sensitivity),'Specificity':make_scorer(specificity),'MCC':make_scorer(MCC)}
kf = StratifiedKFold(n_splits = 10,shuffle=True,random_state = 100)

def build_sklearn_model(X, y, method_name):
    model_map = {"svm":SVC, "knn":KNeighborsClassifier,"nn":MLPClassifier,"rf":RandomForestClassifier}
    tuned_parameters = PARAMETERS[method_name]
    method = model_map[method_name]
    if method == SVC:
        grid = GridSearchCV(method(probability=True,random_state=100), 
                                    param_grid=tuned_parameters,
                                     scoring=SCORING_FNC, cv=kf, n_jobs=N_JOBS, refit='AUC' )
    elif method == KNeighborsClassifier:
        grid = GridSearchCV(method(), param_grid=tuned_parameters, 
                                    scoring =SCORING_FNC, cv=kf, n_jobs=N_JOBS, refit='AUC')
    else:
        grid = GridSearchCV(method(random_state=100), param_grid=tuned_parameters, 
                                    scoring=SCORING_FNC, cv=kf, n_jobs=N_JOBS, refit='AUC')
    grid.fit(X, y)
    return grid.best_estimator_ , grid.best_params_	,grid.cv_results_

def cansmi2fp(cansmi,fpname,json_path = 'FP.json'):
    if isinstance(cansmi,list):
        cansmi = pd.Series(cansmi,name='Canonical_Smiles')
    elif isinstance(cansmi,np.ndarray):
        cansmi = pd.Series(cansmi.flatten(),name='Canonical_Smiles')
    fp_json = pd.read_json(json_path,'index')
    fp_map = cansmi.map(fp_json[fpname])
    converted = []
    for fp in fp_map:
        converted.append([int(i) for i in fp.strip('[]').split(',')])
    return np.array(converted)

def get_performance(cv_res):
    df = pd.DataFrame(cv_res)
    r = df.sort_values(by='rank_test_AUC').iloc[0]
    return r[['mean_test_AUC','mean_test_AUPR', 'mean_test_Accuracy', 'mean_test_BalancedAccuracy','mean_test_Precision','mean_test_Recall','mean_test_F1Score','mean_test_Sensitivity','mean_test_Specificity','mean_test_MCC']]

def check_status(task):
    PASS = False
    if os.path.exists('TaskWatcher.csv.log'):
        AddHeader = False
    else:
        AddHeader = True
        pd.DataFrame([],columns=['task','Status']).set_index('task').to_csv('TaskWatcher.csv.log',mode = 'a',header=AddHeader)  

    df = pd.read_csv('TaskWatcher.csv.log',index_col='task')
    if task in df.index:
        PASS = True
    else:
        pd.DataFrame([[task,'training']],columns=['task','Status']).set_index('task').to_csv('TaskWatcher.csv.log',mode = 'a',header=AddHeader)

    if os.path.exists(f'local/models/{task}.m'):
        df = pd.read_csv('TaskWatcher.csv.log',index_col='task')
        df.at[task,'Status'] = 'Done'
        df.to_csv('TaskWatcher.csv.log')   
    
    return PASS

def plot_roc(model,x_pred,y_true):
    y_score = model.predict_proba(x_pred)[:, 1]
    fpr,tpr,_ = metrics.roc_curve(y_true,y_score)
    auc = metrics.auc(fpr,tpr)
    #disp = plot_roc_curve(model, x_pred, y_true,alpha=0.8)
    return [fpr,tpr,auc]

def plot_pr(model,x_pred,y_true):
    y_score = model.decision_function(x_pred)
    prec,recall,_ = precision_recall_curve(y_true,y_score)
    #disp = plot_precision_recall_curve(model,x_pred,y_true)
    return prec,recall

def CheckHeader(filepath):
    if os.path.exists(filepath):
        AddHeader = False
    else:
        AddHeader = True
    return AddHeader

def local():
    RootDir = os.getcwd()
    DataDir = os.path.join(RootDir,'data')
    name = 'CYP3A4'
    try:
        os.mkdir('local')        
        os.mkdir('local/models')
        os.mkdir('local/temp')
        os.mkdir('data/sampling')
    except Exception:
        pass

    original = pd.read_csv(os.path.join(DataDir,'NewIdea.csv')).rename(columns={'CanonicalSMILES':'Canonical_Smiles'})
    data = original
    unique = data.drop_duplicates('Canonical_Smiles').reset_index(drop = True)
    #prepared = unique.loc[unique.outlier.isna()]
    label = unique['CYP3A4']
    print(name,len(label),Counter(label))
    cansmi = unique['Canonical_Smiles']

    for fpname,method,sampling in product(PADELFPS,METHODS,SAMPLING):
        name_str = f'{name}_{fpname}_{method}_{sampling}'
        
        if not check_status(name_str):
            print(f'ONGOING----> {name_str}')

            # prepare data
            if not os.path.exists(f'Train-{sampling}.csv'):
                if sampling == 'Original':
                    x_resampled,y_resampled = cansmi,label
                else:
                    #x_resampled,y_resampled = Resampling(cansmi2fp(cansmi,fpname,os.path.join(DataDir,'FP.json')),label,sampling)
                    x_resampled,y_resampled = Resampling(np.array(cansmi).reshape(-1,1),label,sampling)
                x_train,x_test,y_train,y_test = train_test_split(x_resampled,y_resampled,test_size = 0.2,random_state = RANDOMSTATE)
                x_test,x_valid,y_test,y_valid = train_test_split(x_test,y_test,test_size = 0.5,random_state = RANDOMSTATE)

                pd.concat([pd.DataFrame(y_train).reset_index(drop=True).rename(columns={0:'CYP3A4'}),pd.DataFrame(x_train).reset_index(drop=True).rename(columns={0:'Canonical_Smiles'})],axis=1).to_csv(os.path.join(DataDir,'sampling',f'Train-{sampling}.csv'),index = False)
                pd.concat([pd.DataFrame(y_test).reset_index(drop=True).rename(columns={0:'CYP3A4'}),pd.DataFrame(x_test).reset_index(drop=True).rename(columns={0:'Canonical_Smiles'})],axis=1).to_csv(os.path.join(DataDir,'sampling',f'Test-{sampling}.csv'),index = False)
                pd.concat([pd.DataFrame(y_valid).reset_index(drop=True).rename(columns={0:'CYP3A4'}),pd.DataFrame(x_valid).reset_index(drop=True).rename(columns={0:'Canonical_Smiles'})],axis=1).to_csv(os.path.join(DataDir,'sampling',f'Valid-{sampling}.csv'),index = False)
            else:
                trainset = pd.read_csv(os.path.join(DataDir,'sampling',f'Train-{sampling}.csv'))
                testset = pd.read_csv(os.path.join(DataDir,'sampling',f'Test-{sampling}.csv'))
                validset = pd.read_csv(os.path.join(DataDir,'sampling',f'Valid-{sampling}.csv'))
                x_train = trainset['Canonical_Smiles']
                x_test = testset['Canonical_Smiles']
                x_valid = validset['Canonical_Smiles']
                y_train = trainset['CYP3A4']
                y_test = testset['CYP3A4']
                y_valid  = validset['CYP3A4']
            x_train = cansmi2fp(x_train,fpname,os.path.join(DataDir,'FP.json'))
            x_test = cansmi2fp(x_test,fpname,os.path.join(DataDir,'FP.json'))
            x_valid = cansmi2fp(x_valid,fpname,os.path.join(DataDir,'FP.json'))
            
            best_model, best_params, cv_res = build_sklearn_model(x_train, y_train, method)
            pd.DataFrame(cv_res).to_csv(f'local/temp/{name_str}_cv.csv',index = False)
            model = best_model
            joblib.dump(model,f'local/models/{name_str}.m')
            TEMP = check_status(name_str)

            AddHeader = CheckHeader(f'local/temp/{name}_train.csv')
            result = [name,fpname,method,sampling,str(best_params)]+get_performance(cv_res).tolist()
            pd.DataFrame([result],columns = ['endpoint','fp','method','sampling','param','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC']).to_csv(f'local/temp/{name}_train.csv',index = False,mode = 'a',header= AddHeader)

            proba = model.predict_proba(x_test)[:,1]
            r = [name,fpname,method,sampling]+list(np.array(list(AllMetrics(y_test,proba).values())).ravel())
            pd.DataFrame([r],columns =['endpoint','fp','method','sampling','Assay','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC','y_true','y_score']).to_csv(f'local/temp/{name}_test.csv',index = False,mode = 'a',header= AddHeader)
            
            validproba = model.predict_proba(x_valid)[:,1]
            r = [name,fpname,method,sampling]+list(np.array(list(AllMetrics(y_valid,validproba).values())).ravel())
            pd.DataFrame([r],columns =['endpoint','fp','method','sampling','Assay','AUC','AUPR','Accuracy','BalancedAccuracy','Precision','Recall','F1Score','Sensitivity','Specificity','MCC','y_true','y_score']).to_csv(f'local/temp/{name}_valid.csv',index = False,mode = 'a',header= AddHeader)
    
if __name__ == "__main__":
    
    local()  ### sklearn import multiprocessing which means only by run code under __main__ part(/freeze_support) can success to finish
    #JobDone('try')