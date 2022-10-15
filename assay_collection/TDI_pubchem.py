import multiprocessing as mp
import pandas as pd
import os
import requests
import time
import random

def Download(url,file_name):
    s = requests.Session()
    time.sleep(random.randint(1,5))
    try:
        response = s.get(url,stream=True)
        with open(file_name,'wb') as f:
            for item in response.iter_content(chunk_size=512):
                if item:
                    f.write(item)

    except:
        Download(url,file_name)

def Get_urls(file):
    AIDS = pd.read_csv(file)['aid']  # input your own aid column of the table
    AIDS = list(AIDS.dropna().apply(lambda x:int(x)))
    url_list =['https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{}/CSV'.format(i) for i in AIDS]
    return url_list  

def main(file):
    import os 
    try: 
        os.mkdir('temp')
    except:
        pass
    url_list = Get_urls(file)
    pool = mp.Pool()
    for url in url_list:
        file_name = r'./temp/{}.csv'.format(url.split('/')[-2])
        print(file_name)
        pool.apply_async(Download,args =(url,file_name))
    pool.close()
    pool.join()
    print ('all download finished')

if '__main__'  ==__name__:
	main(r"tdi.csv") 
