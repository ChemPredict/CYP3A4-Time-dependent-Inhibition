
import pandas as pd 
import os
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids,NeighbourhoodCleaningRule,NearMiss
global RANDOMSTATE
RANDOMSTATE = 1375
np.random.seed(RANDOMSTATE)

def Resampling(x,y,sampling):
    'x:cansmi y:label, type = pd.Series'
    SamplingMap = {
        'ROS':RandomOverSampler,'SMOTE':SMOTE,'ADASYN':ADASYN,'RUS':RandomUnderSampler,'CC':ClusterCentroids,'NCR':NeighbourhoodCleaningRule,'NM':NearMiss
    }
    if sampling == 'NM' or 'NCR':
        x_resampled,y_resampled = SamplingMap[sampling]().fit_resample(x,y)
    else:
        x_resampled,y_resampled = SamplingMap[sampling](random_state = RANDOMSTATE).fit_resample(x,y)
    print(f'Resampled via [{sampling}] | {Counter(y_resampled).items()}')
    return x_resampled,y_resampled.to_numpy()


'''

        idx = x.index.to_numpy().reshape(-1,1)
        print(x)
        x_resampled,y_resampled = SamplingMap[sampling](random_state = RANDOMSTATE).fit_resample(idx,y)
        print(x_resampled)
        print(f'Resampled via [{sampling}] | {Counter(y_resampled).items()}')
        x_resampled = pd.DataFrame(x_resampled)[0].map(x).to_numpy().reshape(-1,1)
        y_resampled = y_resampled.to_numpy().reshape(-1,1)
    return x_resampled,y_resampled'''