#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import numpy as np
import faiss
import pandas as pd
from collections import Counter
import sys
import argparse
import joblib
from joblib import Parallel,delayed
from joblib import parallel_backend
from pandarallel import pandarallel
from  tqdm import tqdm
pandarallel.initialize()
import json
from datetime import datetime
from datetime import timedelta
import pickle

# ---------------------------------------------
USE_TFIDF = True
USE_doc2vec = True
model_pkl_dir = 'model_pkl_dir'
mapping_df_dir = 'mapping_data_dir'

# ---------------------------------------------
def read_vectors_from_file(_typeID, _date):
    global model_pkl_dir
    if _typeID == 'LongFormer':
        fname = os.path.join(model_pkl_dir, "doc_id2sBertEmb_{}.pkl".format(_date))
    elif _typeID == 'sBert':
        fname = os.path.join(model_pkl_dir,"doc_id2LongFormerEmb_{}.pkl".format(_date))
    elif _typeID == 'tfidf':
        fname =  os.path.join(model_pkl_dir, "doc_id2tfidfEmb_{}.pkl".format(_date))
    elif _typeID == 'doc2vec':
        fname =  os.path.join(model_pkl_dir, "doc_id2doc2vecEmb_{}.pkl".format(_date))
       
    with open(fname,'rb') as fh:
        vec = pickle.load(fh)
        vec = np.array(list(vec.values()))
        return vec

        
def read_indices_from_file( _typeID, date):
    global model_pkl_dir
    filename = os.path.join(model_pkl_dir ,'faiss_index_{}_{}'.format(_typeID, date))
    index = faiss.read_index(filename)
    return index
    
    
class query_doc:
    def __init__(self, date_str, nprobe=25):
        self.date_str = date_str
        self.nprobe = nprobe
        self.df_Mapping = pd.read_csv(os.path.join(mapping_df_dir, 'mapping_data_{}.csv'.format(date_str)),index_col=None)
        
        self.index_tfidf = None
        self.index_doc2vec = None

        self.vectors_tfidf = None 
        self.vectors_doc2vec = None
        
        self.index_tfidf = read_indices_from_file('tfidf', date_str)    
        self.index_tfidf.nprobe = nprobe

        self.index_doc2vec = read_indices_from_file('doc2vec',date_str)
        self.index_doc2vec.nprobe = nprobe
        
        self.vectors_tfidf = read_vectors_from_file('tfidf', date_str)  
        self.vectors_doc2vec = read_vectors_from_file('doc2vec', date_str)  
        
        return

        
    def query(
        self,
        doc_ID = None, 
        synID = None,
        find_NN = 10,
        min_count_threshold = 2,
        n_probe = 20
    ):
        obj_list = [self.index_tfidf, self.index_doc2vec]
        vec_list = [self.vectors_tfidf, self.vectors_doc2vec]
        result = []

        if doc_ID is None and synID is None :
            return
        if doc_ID is not None:
            _tmp_ = df_Mapping.loc[(df_Mapping['id']==doc_ID)]
            synID = _tmp_['synID'].values[0]

        _type_of_index = ['tfidf', 'doc2vec']
        i = 0 
        for _index,_vector in zip(obj_list, vec_list):
            i+=1
            if _index is None: continue
            _index.nprobe = n_probe
            D, I = _index.search(
                np.array([_vector[synID]]).astype(np.float32),
                find_NN
            ) 
            result.extend(I[0][1:])

        counter = Counter(result)
        filtered = [ k for k,v in counter.items() if v >= min_count_threshold and k >-1 and k!=synID]    
        return filtered

def test(input_syn_id=32):
    query_obj = query_doc('2020-12-10')
    query_obj.query( synID = 32 , min_count_threshold = 2)
   
    res = query_obj.query(
        synID = input_syn_id, 
        find_NN = 10,
        min_count_threshold = 2,
        n_probe = 20
    )
    print(input_syn_id, res)
    for r in [input_syn_id] + res:
        print(query_obj.df_Mapping.loc[query_obj.df_Mapping['synID']==r].title)
    return   

test(250)

test(32)


# In[ ]:




