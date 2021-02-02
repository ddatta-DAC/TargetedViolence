#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import os

import sBert
import doc2vec
import pandas as pd
import sBert
import LongFormer

import os
import sys
import faiss
import numpy as np
import multiprocessing as mp
import joblib
from joblib import Parallel,delayed
from  tqdm import tqdm
from pathlib import Path
import glob
import json
from joblib import parallel_backend
import datetime
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
SPACY_tokenizer = Tokenizer(nlp.vocab)
import argparse
import spacy
import sys
import faiss
import numpy as np
import multiprocessing as mp
import glob
import re
from joblib import parallel_backend
import datetime
import pickle
import json
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, strip_accents_ascii
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import Parallel,delayed 
import multiprocessing as mp
from collections import OrderedDict
import faiss
from datetime import datetime
from datetime import timedelta
from sklearn.decomposition import TruncatedSVD
from time import time
from pathlib import Path
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
SPACY_tokenizer = Tokenizer(nlp.vocab)
from nltk.corpus import stopwords
nltk_stop_words = stopwords.words('english')
from pandarallel import pandarallel
from  tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
pandarallel.initialize()
from collections import OrderedDict
SPACY_NLP = spacy.load('en_core_web_lg')

# Editable location 
# =======================================================================================================================
from os.path import expanduser
home = expanduser("~")
DATA_DIR = os.path.join(home, 'hatespeech_prod_data','data', 'processed')
model_pkl_dir = 'model_pkl_dir'
REFRESH_INDEX = True
# ===================================================================================================================================


def custom_tokenizer_function (x):
    global SPACY_tokenizer
    global nltk_stop_words
    doc = SPACY_tokenizer.__call__(x)
    tokens = [str.lower(d.lemma_)  for d in doc ]
    tokens = [ w for w in tokens if w not in nltk_stop_words and len(w)>1]
    return tokens

# ------------------------
# Remove stopwords 
# ------------------------
def clean_text_1(text):
    global nltk_stop_words
    remove_chars = ['\n','\t','!',',',';', ':', '?']
    for c in remove_chars: text = text.replace(c,' ')
    words = text.split(' ')
    words = [w for w in words if not w in nltk_stop_words]
    text = ' '.join(words)
    text = ' '.join(custom_tokenizer_function (text))
    return text


def clean_text_3(text):
    global nltk_stop_words
    global SPACY_NLP
    global SPACY_tokenizer
    text = re.sub('[0-9]+', '', text)
    sentences = [s.__repr__().strip() for s in SPACY_NLP(text).sents][:10]
    sentences = [  SPACY_tokenizer.__call__(s) for s in sentences]
    
    s1 = [ [ str.lower(w.lemma_)  for w in  sent if len(w.lemma_)>1 and w.lemma_ not in nltk_stop_words ] for sent in sentences]
    text = []
    for _ in s1: text.extend(_)
    return text


# ------------------------
# Not Remove stopwords 
# ------------------------
def clean_text_2(text):
    global SPACY_NLP
    remove_chars = ['\n','\t','!',',',';', ':']
    for c in remove_chars: 
        text = text.replace(c,' ')
    sentences = [s.__repr__().strip() for s in SPACY_NLP(text).sents][:5]
    text = ' '.join(sentences)
    return text

def get_doc_id(doc_dict):
    return doc_dict.get('id')

'''
Save all resuts in a dataframe
'''

def process_doc(doc_dict):
    res = {}
    res['id'] = get_doc_id(doc_dict)
    res['title'] = doc_dict['title']
    res['xml_date'] = doc_dict['xml_date']
    res['text'] = doc_dict['text']
    res['complexId'] =  str(res['id']) + '_' + str(res['xml_date']) 
    return res

def process_file(_file):
    arr_docs = []
    fh = open(_file,'r')

    def _aux_(l):
        try:
            _doc = json.loads(l)
            return _doc
        except:
            pass 
    result = Parallel(n_jobs= mp.cpu_count())(delayed(_aux_)(l)  for l in fh.readlines() )    
    fh.close() 
    
    result = Parallel(n_jobs= mp.cpu_count())(delayed(process_doc)(_doc)  for _doc in result if _doc is not None) 
    filtered_results = []
    for r in result:
        r1 = r.copy()
        r1['path']=_file
        filtered_results.append(r1)
    return filtered_results

    
def aux_process_subdir(sub_dir):
    folder_files = glob.glob(os.path.join(sub_dir,'**.json'))
    results = []
    for file in folder_files:
        res = process_file(file)
        results.extend(res)
    return results




# ==========
# Format '2020-10-05'
# ==========
def get_date_range(_date_str, diff  ):
    valid_dates = [(datetime.fromisoformat(_date_str) + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(-diff, diff+1)]
    return valid_dates

def get_candidateDup_dateDirs(_dir_, diff = 5):
    global DATA_DIR     
    date_str = _dir_.split('/')[-1]
    try:
        dt_obj = datetime.fromisoformat(date_str)
    except:
        return []
    valid_dates = get_date_range(date_str, diff = diff )
    _parent = _dir_.replace(date_str,'')
    valid_dirs = [ os.path.join(_parent, vd) for vd in valid_dates]
    valid_dirs = [ _ for _ in valid_dirs if  os.path.exists(_)]
    return valid_dirs

# -------------------
# Set up
# 1. create emebedding(s) for each doc
# 2. Primary key (id + xml_date)
# 3. synthetic ID for each (+/-7 day period)
# -------------------

def process_all_files(dir_list):
    # -----------------------------------
    # Assign each to a separate thread
    # -----------------------------------
    results = []
    collated = []
    for i in tqdm(range(len(dir_list))):
        r = aux_process_subdir(dir_list[i])
        collated.extend(r)
    
    synID = 0 
    for i in range(len(collated)):
        collated[i]['synID'] = synID
        synID += 1
    return collated





def obtain_value_list(list_data, key):
    return [item[key] for item in list_data]

def get_sBertEmbedding( list_dataDict ):
    _text = obtain_value_list(list_dataDict, key='text') 
    _synID = obtain_value_list(list_dataDict, key='synID')
    
    # Preprocess the data for tokenizer
    def aux_sb1( _id, _txt ):
        return (_id, clean_text_2(_txt))
    
    def aux_sb2(_id, _txt ):
        return (_id, sBert.get_doc_emb(_txt))
    
    results = []
    with parallel_backend('threading', n_jobs=100):
        results = Parallel()(delayed(aux_sb1)( _id,_txt)  for _id,_txt in zip(tqdm(_synID), _text) )
    
    results = Parallel(
        n_jobs = 10
    )(delayed(aux_sb2)( _id_txt[0],_id_txt[1]) for _id_txt in tqdm(results))
    
    results = OrderedDict({ item[0] : item[1] for item in results })
    return results
    

def get_LongFormerEmbedding( list_dataDict ):
    
    _text = obtain_value_list(list_dataDict, key='text') 
    _synID = obtain_value_list(list_dataDict, key='synID')
    
    # Preprocess the data for tokenizer
    def aux_sb1( _id, _txt ):
        return (_id, clean_text_2(_txt))
    
    def aux_sb2(_id, _txt ):
        return (_id, LongFormer.get_doc_emb(_txt))
    
    results = []
    with parallel_backend('threading', n_jobs=100):
        results = Parallel()(delayed(aux_sb1)( _id,_txt)  for _id,_txt in zip(tqdm(_synID), _text) )
    
    results = Parallel(
        n_jobs = 10
    )(delayed(aux_sb2)( _id_txt[0],_id_txt[1]) for _id_txt in tqdm(results))
    
    results = OrderedDict({ item[0] : item[1] for item in results })
    
    return results

def get_tfidfEmb(list_dataDict):
    
    kwvectorizer = TfidfVectorizer(
        strip_accents='unicode',
        ngram_range=(1, 1), 
        stop_words='english', 
        min_df=0.01, 
        max_df=0.90
    )

    _text = obtain_value_list(list_dataDict, key='text') 
    _synID = obtain_value_list(list_dataDict, key='synID')

    def aux_1( _id, _txt ):
        return (_id, clean_text_1(_txt))

    results = []
    with parallel_backend('threading', n_jobs=100):
        results = Parallel()(delayed(aux_1)( _id,_txt)  for _id,_txt in zip(tqdm(_synID), _text) )


    doc_list = [ _item[1]  for _item in results ]
    id_list = [_item[0] for _item in results]

    xformed_docs = kwvectorizer.fit_transform(doc_list)
    svd_obj = TruncatedSVD(n_components=256)
    svd_obj.fit(xformed_docs.todense())
    vec = svd_obj.transform(xformed_docs.todense())
    tfidfEmb = OrderedDict({
        _id: _vec for _id,_vec in zip(id_list,vec)
    })
    return tfidfEmb
    

def get_doc2vecEmb(list_dataDict):

    _text = obtain_value_list(list_dataDict, key='text') 
    _synID = obtain_value_list(list_dataDict, key='synID')

    def aux_1( _id, _txt ):
        return (_id, clean_text_3(_txt))

    results = []
    with parallel_backend('threading', n_jobs=100):
        results = Parallel()(delayed(aux_1)( _id,_txt)  for _id,_txt in zip(tqdm(_synID), _text) )
    
    id_list = [_item[0] for _item in results]
    doc_list = [ _item[1]  for _item in results ]
   
    
    vectors, model = doc2vec.build_model(id_list, doc_list, epochs=50)
    doc2vecEmb = OrderedDict({
        _id: _vec for _id,_vec in zip(id_list, vectors)
    })
    return doc2vecEmb
        
def set_up_index(
    docId_list,
    vectors, 
    _typeID
):
    global model_pkl_dir
    global REFRESH_INDEX
    filename = os.path.join ( model_pkl_dir, 'faiss_index_{}'.format(_typeID))
    if REFRESH_INDEX is False and os.path.exists(filename):
        index = faiss.read_index(filename)
        return index
    
    input_count = vectors.shape[0]                  
    d = vectors.shape[1]                            
    
    M = 4  # The number of sub-vector. Typically this is 8, 16, 32, etc.
    nbits = 4 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
    # Param of IVF
    nlist = 100  # The number of cells (space partition). Typical value is sqrt(N)
    # Param of HNSW
    # The number of neighbors for HNSW. This is typically 32
    hnsw_m = 16  
    
    # Setup
    quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
    index = faiss.IndexIVFPQ(quantizer,d, nlist, M, nbits)

#     nlist = 100
#     k = 4
#     quantizer = faiss.IndexFlatL2(d)  # the other index
#     index = faiss.IndexIVFFlat(quantizer, d, nlist)
 
    t0 = time()
    index.train(vectors)
    t1 = time()
    print('Is Index Trained ?', index.is_trained)
    t2 = time() 
    index.add_with_ids(
        vectors, 
        np.array(docId_list).astype(np.int) 
    )
    t3 = time()
    print('Time taken for Train {:.5f} ||  Adding index {:.5f}'.format(t1-t0, t3-t2))
    
    # -----------------
    # Save index 
    # -----------------
    faiss.write_index(index, filename)
    return index



def setUp_index_byDate(date_str = None,  max_date_diff = 2):
    global DATA_DIR
    global model_pkl_dir
    
    mapping_df_dir = 'mappind_data_dir'
    pathobj = Path(mapping_df_dir)
    pathobj.mkdir(exist_ok=True,parents=True)
    
    pathobj = Path(model_pkl_dir)
    pathobj.mkdir(exist_ok=True,parents=True)
    if date_str is None:
        return
    date = date_str
    list_sub_dirs = sorted([f.path for f in os.scandir(DATA_DIR) if f.is_dir()])
    candidate_dirs = get_candidateDup_dateDirs( os.path.join(DATA_DIR, "{}".format(date)), diff = max_date_diff)


    preProc_dataFile =  os.path.join( model_pkl_dir, 'preProc_dataFile_{}.pkl'.format(date) )
    print(preProc_dataFile)
    if os.path.exists(preProc_dataFile):
        with open(preProc_dataFile,'rb') as fh:
            results = pickle.load(fh)
    else:
        # Obtain and save results
        results = process_all_files(candidate_dirs)
        with open(preProc_dataFile,'wb') as fh:
            pickle.dump(results,fh, pickle.HIGHEST_PROTOCOL)
    collated_data = results

    # collated_data = collated_data[:500]

    savefile_name_sBertEmbedding = os.path.join( model_pkl_dir, 'doc_id2sBertEmb_{}.pkl'.format(date))
    savefile_name_LongFormerEmbedding = os.path.join( model_pkl_dir,'doc_id2LongFormerEmb_{}.pkl'.format(date))
    savefile_name_tfidfEmbedding =  os.path.join( model_pkl_dir, 'doc_id2tfidfEmb_{}.pkl'.format(date))
    savefile_name_doc2vecEmbedding =  os.path.join( model_pkl_dir, 'doc_id2doc2vecEmb_{}.pkl'.format(date))   
    
    if os.path.exists(savefile_name_sBertEmbedding):
        with open(savefile_name_sBertEmbedding,'rb') as fh:
            sBertEmbedding = pickle.load(fh)
    else:
        sBertEmbedding = OrderedDict({})
        # sBertEmbedding = get_sBertEmbedding( collated_data )
        with open(savefile_name_sBertEmbedding,'wb') as fh:
            pickle.dump( sBertEmbedding, fh, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(savefile_name_LongFormerEmbedding):
        with open(savefile_name_LongFormerEmbedding,'rb') as fh:
            LongFormerEmbedding = pickle.load(fh)
    else:
        LongFormerEmbedding = OrderedDict({})
        # LongFormerEmbedding = get_LongFormerEmbedding(collated_data )
        with open(savefile_name_LongFormerEmbedding,'wb') as fh:
            pickle.dump( LongFormerEmbedding, fh, pickle.HIGHEST_PROTOCOL)
    
    if os.path.exists(savefile_name_doc2vecEmbedding):
        with open(savefile_name_doc2vecEmbedding,'rb') as fh:
            doc2vecEmbedding = pickle.load(fh)
    else:
        doc2vecEmbedding = get_doc2vecEmb( collated_data )
        with open(savefile_name_doc2vecEmbedding,'wb') as fh:
            pickle.dump( doc2vecEmbedding, fh, pickle.HIGHEST_PROTOCOL)

    
    if os.path.exists(savefile_name_tfidfEmbedding):
        with open(savefile_name_tfidfEmbedding,'rb') as fh:
            doc_id2tfidfEmb = pickle.load(fh)
    else:
        doc_id2tfidfEmb =  get_tfidfEmb(collated_data)
        with open(savefile_name_tfidfEmbedding,'wb') as fh:
            pickle.dump( doc_id2tfidfEmb, fh, pickle.HIGHEST_PROTOCOL)
    
    
    # Create mapping 
    synID = obtain_value_list(collated_data, 'synID') 
    _id = obtain_value_list(collated_data, 'id') 
    _path =  obtain_value_list(collated_data, 'path')
    _title = obtain_value_list(collated_data, 'title')
    xml_date = obtain_value_list(collated_data, 'xml_date') 
    
    df_Mapping = pd.DataFrame({'synID':synID , 'id': _id, 'path': _path, 'title': _title, 'xml_date': xml_date})
    df_Mapping.to_csv('mapping_data.csv',index=None)

    df_Mapping.to_csv(
        os.path.join(mapping_df_dir, 'mapping_data_{}.csv'.format(date_str)),index=None
    )
    # Create mapping
    docId_list = obtain_value_list(collated_data,'synID')
    vectors_text_tfidf = np.array(list(doc_id2tfidfEmb.values())).astype(np.float32)
    vectors_text_sBert = np.array(list(sBertEmbedding.values())).astype(np.float32)
    vectors_text_LFormer = np.array(list(LongFormerEmbedding.values())).astype(np.float32)
    vectors_text_doc2vec = np.array(list(doc2vecEmbedding.values())).astype(np.float32)
 
    
    index_tfid = set_up_index(
            docId_list = docId_list,
            vectors= vectors_text_tfidf,
            _typeID ='tfidf_{}'.format(date)
    )
    index_doc2vec = set_up_index(
        docId_list = docId_list,
        vectors= vectors_text_doc2vec,
        _typeID ='doc2vec_{}'.format(date)
    )
    
#     try:
#         index_tfid = set_up_index(
#             docId_list = docId_list,
#             vectors= vectors_text_tfidf,
#             _typeID ='tfidf_{}'.format(date)
#         )
#     except:
#         print('ERROR :: tfidf index')
#     try:
        
#         index_sBert = set_up_index(
#             docId_list = docId_list,
#             vectors= vectors_text_sBert,
#             _typeID ='sBert_{}'.format(date)
#         )
#     except:
#         print('ERROR :: sBert index')
#     try:
#         index_LFormer = set_up_index(
#             docId_list = docId_list,
#             vectors= vectors_text_LFormer,
#             _typeID ='LongFormer_{}'.format(date)
#         )
#     except:
#         print('ERROR :: longformer index')
#     try:
#         index_doc2vec = set_up_index(
#             docId_list = docId_list,
#             vectors= vectors_text_doc2vec,
#             _typeID ='doc2vec_{}'.format(date)
#         )
#     except:
#         print('ERROR :: doc2vec index')
    
    return

                                    
# -----------------------------------------------
# Function to set up all indices
# -----------------------------------------------                                    
def setup_all():
    global DATA_DIR
    list_sub_dirs = sorted([f.path for f in os.scandir(DATA_DIR) if f.is_dir()])
    
    valid_dates = []
    for _subdir in list_sub_dirs:
        date_str = _subdir.split('/')[-1]
        try:
            dt_obj = datetime.fromisoformat(date_str)
        except:
            continue
        __date_str__= dt_obj.strftime('%Y-%m-%d')
        valid_dates.append(__date_str__)
    valid_dates = sorted(valid_dates)
    
    for _date_str in valid_dates[-60:]:
        setUp_index_byDate(_date_str)
    return

# setUp_index_byDate(date_str = '2020-12-10',  max_date_diff = 2)
# setUp_index_byDate(date_str = '2021-01-01',  max_date_diff = 2)
# setUp_index_byDate(date_str = '2021-01-04',  max_date_diff = 2)
# setUp_index_byDate(date_str = '2021-01-05',  max_date_diff = 2)
# setUp_index_byDate(date_str = '2021-01-09',  max_date_diff = 2)


setup_all()


