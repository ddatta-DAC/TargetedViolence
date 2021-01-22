#!/usr/bin/env python
# coding: utf-8

# In[146]:


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



# ===========================================================================================
# Should run Indexing.py first
# 
# =================================================================
# Editable settings
# =================================================================================================
USE_TFIDF = True
USE_sBert = True
USE_Longformer = False
N_PROBE = 100
# =================================================================================================


def read_indices_from_file(_typeID):
    filename = 'faiss_index_{}'.format(_typeID)
    index = faiss.read_index(filename)
    return index

def read_vectors_from_file(_typeID):
    if _typeID == 'Longformer':
        fname = 'doc_id2LongformerEmb.pkl'
    if _typeID == 'sBert':
        fname = 'doc_id2BertEmb.pkl'
    if _typeID == 'tfidf':
        fname = 'doc_id2textSvdEmb.pkl'
    with open(fname,'rb') as fh:
        vec = pickle.load(fh)
        return vec
    
def query(
    doc_ID, 
    xml_date = None,
    find_NN = 20,
    min_count_threshold = 1,
    check_date = True,
    max_date_diff = 2
):
    global df_Mapping
    global index_sBert
    global index_tfidf
    global index_longformer
    global vectors_longformer
    global vectors_sBert
    global vectors_tfidf
   
    if xml_date is not None:
        _tmp_ = df_Mapping.loc[(df_Mapping['xml_date']==xml_date)&(df_Mapping['id']==doc_ID)]
        synID = _tmp_.iloc[0]['synID'].values[0]
        xml_date = _tmp_['xml_date'].values[0]
    else:
        _tmp_ = df_Mapping.loc[(df_Mapping['id']==doc_ID)]
        synID = _tmp_['synID'].values[0]
        xml_date = _tmp_['xml_date'].values[0]

    obj_list = [index_tfidf, index_sBert, index_longformer]
    vec_list = [vectors_tfidf, vectors_sBert, vectors_longformer]
    result = []
    
    _type_of_index = ['tfidf', 'bert', 'lf']
    i = 0 
    for _index,_vector in zip(obj_list, vec_list):
        i+=1
        if _index is None: continue
        D, I = _index.search(
            np.array( [_vector[synID]]).astype(np.float32),
            find_NN
        ) 
        result.extend(I[0][1:])
        
    counter = Counter(result)
    filtered_1 = [ k for k,v in counter.items() if v >= min_count_threshold]
    cur_doc_date = get_date(synID)
    
    if check_date:
        filtered_2 = []
        for _synID in filtered_1: 
            _date = get_date(_synID)
            delta = cur_doc_date - _date
            if abs(delta.days) < max_date_diff:
                filtered_2.append(_synID)
    else:
        filtered_2 = filtered_1
    return filtered_2




def get_date(synID):
    global df_Mapping
    date_str = df_Mapping.loc[(df_Mapping['synID']==synID)]['xml_date'].values[0]
    date_str = date_str.split('T')[0]
    return datetime.fromisoformat(date_str)

get_date(10)

df_Mapping = pd.read_csv('mapping_data.csv',index_col=None)
_typeID = ['tfidf', 'sBert', 'Longformer']


index_tfidf = None
index_sBert = None
index_longformer = None
vectors_longformer = None
vectors_sBert = None
vectors_tfidf = None

if USE_sBert:
    index_sBert = read_indices_from_file('sBert')
    index_sBert.nprobe = N_PROBE
if USE_TFIDF:
    index_tfidf = read_indices_from_file('tfidf')
    index_tfidf.nprobe = N_PROBE
if USE_Longformer:
    index_longformer= read_indices_from_file('Longformer')
    index_longformer.nprobe = N_PROBE
    

try:
    vectors_longformer = read_vectors_from_file('Longformer')
except:
    print('error reading vectors')
try:    
    vectors_sBert = read_vectors_from_file('sBert')
except:
    print('error reading vectors')

try:
    vectors_tfidf = read_vectors_from_file('tfidf')
except:
    print('error reading vectors')


def obtain_details_from_synID(synID):
    global df_Mapping
    __tmp__ = df_Mapping.loc[(df_Mapping['synID']==synID)]
    fpath = __tmp__['path'].values[0]
    _id = __tmp__['path'].values[0]
    _date = __tmp__['xml_date'].values[0]
    return _id, _date, fpath





# -----------------------
## Example Query
# -----------------------
result = query(
    doc_ID ='006EEC7077F095B8BF022A47340EC9AC', 
    xml_date = None,
    find_NN = 100,
    min_count_threshold = 2,
    check_date=True
)
print(obtain_details_from_synID(113), obtain_details_from_synID(25), obtain_details_from_synID(226) )
















# In[ ]:




