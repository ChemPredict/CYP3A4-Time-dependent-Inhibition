import os 
import pandas as pd 
import numpy as np

merged = pd.DataFrame()
for filename in os.listdir('temp'):
    path = os.path.join(os.getcwd(),'temp',filename)
    with open(path,'r') as f:
        contents = f.readlines()
        f.close()
    if 'Status' in contents[0]:
        with open('failed.txt','a') as fail:
            fail.write(f"{filename.split('/')[-1].strip('.csv')},{''.join(contents)},\n")
            fail.close()
    else:
        df = pd.read_csv(path)
        merged = pd.concat([merged,df])

merged.to_csv('merged.csv')