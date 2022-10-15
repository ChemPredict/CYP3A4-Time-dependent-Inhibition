#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


# here put the import lib
import os,time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import string
from matplotlib import gridspec
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
fontsize = 100
sns.set_theme(style='whitegrid')
FontDict = {
    'family':'Times New Roman',
    'weight':'normal',
    'size': fontsize,
}
c1 = '#1f77b4'
c2 = '#ff7f0e'
DescriptorsDict ={
            'Molecular Weight': Descriptors.ExactMolWt,
            'LogP': Descriptors.MolLogP,
            'Number of H Acceptors': Descriptors.NumHAcceptors,
            'Number of H Donors': Descriptors.NumHDonors,
            'Topological Polar Surface Area(TPSA)': Chem.rdMolDescriptors.CalcTPSA,
        }
for idx,dataset in enumerate(['RUS','Original'][::-1]):
    DataDir = '../sampling'
    figure = plt.figure(figsize=(100,100))
    gs = gridspec.GridSpec(3,2)
    gs.update(hspace = 0.2)
    
#dataset = 'RUS'
    table = pd.DataFrame()
    for filename in os.listdir(DataDir):
        if dataset in filename:
            temp = pd.read_csv(os.path.join(DataDir,filename))
            temp['label'] = filename.split('-')[0]
            table = pd.concat([table,temp],axis = 0)
    table = table.reset_index(drop=True)
        
    for idx,descriptor in enumerate(DescriptorsDict.keys()):
        table[descriptor] = table['Canonical_Smiles'].apply(lambda x: DescriptorsDict[descriptor](Chem.MolFromSmiles(x)))
        ax = plt.subplot(gs[idx//2, idx%2])
        print(f'ploting {idx//2, idx%2}')
        sns.distplot(table.loc[table['CYP3A4']==1,descriptor],label='TDI',color=c1,ax = ax, kde=False)
        ax.axvline(table.loc[table['CYP3A4']==1,descriptor].mean(),color=c1,linestyle='--',linewidth = 10)
        sns.distplot(table.loc[table['CYP3A4']==0,descriptor],label='Non-TDI',color=c2, ax = ax, kde=False)
        ax.axvline(table.loc[table['CYP3A4']==0,descriptor].mean(),color=c2,linestyle='--',linewidth = 10)
        ax.set_xlabel (descriptor,FontDict)
        ax.set_ylabel ('Density',FontDict)
        ax.set_title(f'{string.ascii_uppercase[idx]})',FontDict, fontweight = 'bold',loc='left', fontsize = 1.25*fontsize, y = 1.05,x = -0.15)
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.legend(title='Compound Group',prop = FontDict, title_fontsize = fontsize)
        plt.tick_params('both',width=2,labelsize=fontsize)
    descriptors = table.loc[:, DescriptorsDict.keys()].values
    descriptors_std = StandardScaler().fit_transform(descriptors)
    pca = PCA()
    descriptors_2d = pca.fit_transform(descriptors_std)
    descriptors_pca= pd.DataFrame(descriptors_2d)
    descriptors_pca.index = table.index
    descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    # This normalization will be performed just for PC1 and PC2, but can be done for all the components.
    scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
    scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

    # And we add the new values to our PCA table
    descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
    descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]

    descriptors_pca.to_csv(f'{dataset}_pca.csv')


    
    ax = plt.subplot(gs[-1,-1])
    sns.scatterplot(descriptors_pca.loc[table['CYP3A4']==1,'PC1_normalized'],descriptors_pca.loc[table['CYP3A4']==1,'PC2_normalized'],color=c1,label='TDI',ax = ax,s = 500)
    sns.scatterplot(descriptors_pca.loc[table['CYP3A4']==0,'PC1_normalized'],descriptors_pca.loc[table['CYP3A4']==0,'PC2_normalized'],color=c2,label='Non-TDI',ax = ax, s = 500)
    ax.set_xlabel ('PC1',FontDict)
    ax.set_ylabel ('PC2',FontDict)
    ax.set_title(f'{string.ascii_uppercase[len(DescriptorsDict)]})',FontDict, fontweight = 'bold',loc='left', fontsize=1.25*fontsize, y = 1.05,x = -0.15)
    plt.legend(title='Compound Group',prop = FontDict,title_fontsize = fontsize, markerscale = 3)
    ax.axhline(0, color='black',linestyle='--')
    ax.axvline(0, color='black',linestyle='--')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.tick_params('both',width=2,labelsize=0.8*fontsize)
    #plt.tight_layout()
    plt.savefig(f'{dataset}_ChemicalProfile.png',dpi=300, bbox_inches='tight')
#plt.legend(loc = 'upper right',FontDict,bbox_to_anchor=(1.175,1.1))
#plt.savefig('PCAwithdataset.png',dpi=300)