#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
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
import json
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
    text = custom_tokenizer_function (text)
    return text

# ------------------------
# Not Remove stopwords 
# ------------------------
def clean_text_2(text):
    global SPACY_NLP
    remove_chars = ['\n','\t','!',',',';', ':']
    for c in remove_chars: 
        text = text.replace(c,' ')
    sentences = [s.__repr__().strip() for s in SPACY_NLP(text).sents][:10]
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
    res['xml_date'] = doc_dict['xml_date']
    res['text_no_SW'] = clean_text_1(doc_dict['text'])
    res['text_w_SW'] = clean_text_2(doc_dict['text'])
    res['complexId'] =  str(res['id']) + '_' + str(res['xml_date']) 
    return res

def process_file(_file):
    arr_docs = []
    fh = open(_file,'r')
    for l in fh.readlines():
        try:
            _doc = json.loads(l)
            arr_docs.append(_doc)
        except:
            pass
    fh.close()    
    result = Parallel(n_jobs= mp.cpu_count())(delayed(process_doc)(_doc)  for _doc in arr_docs) 
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

def process_all_files(dir_loc):
    list_sub_dirs = sorted([f.path for f in os.scandir(dir_loc) if f.is_dir()])
    # -----------------------------------
    # Assign each to a separate thread
    # ------------------------------------
    results = []
    with parallel_backend('threading', n_jobs=40):
        results = Parallel()(delayed(aux_process_subdir)(list_sub_dirs[i])  for i in tqdm(range(len(list_sub_dirs))) )
    
    collated = []
    for r in results:
        collated.extend(r) 
    
    synID = 0 
    for i in range(len(collated)):
        collated[i]['synID'] = synID
        synID += 1
    return collated



preProc_dataFile = 'preProc_dataFile.pkl'
if os.path.exists(preProc_dataFile):
    with open(preProc_dataFile,'rb') as fh:
        results = pickle.load(fh)
else:
    # Obtain and save results
    results = process_all_files(DATA_DIR)
    with open(preProc_dataFile,'wb') as fh:
        pickle.dump(results,fh, pickle.HIGHEST_PROTOCOL)
        
        
# df = pd.DataFrame(results)  
# df.to_csv('tmp.csv',index=None)

collated_data = results

sBERT_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
sBert_model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
lFormer_tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
lFormer_model = AutoModel.from_pretrained("allenai/longformer-base-4096")

# -----------------------------
# sentences should be a string 
# -----------------------------
def get_sentence_embedding_sBert(sentences):
    global sBERT_tokenizer
    global sBert_model
    if len(sentences) == 0 : return None
    encoded_input = sBERT_tokenizer(sentences, padding=True , truncation=True, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = sBert_model(**encoded_input)
        sentence_embeddings = model_output[0][:,0] #Take the first token ([CLS]) from each sentence 

    return sentence_embeddings.cpu().data.numpy()

def get_sentence_embedding_Longformer(sentences):
    global lFormer_model
    global lFormer_tokenizer
    if len(sentences) == 0 : return None
    encoded_input = lFormer_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = lFormer_model(**encoded_input)
        sentence_embeddings = model_output[0][:,0] #Take the first token ([CLS]) from each sentence 

    return sentence_embeddings.cpu().data.numpy()

def obtain_value_list(list_data, key):
    return [item[key] for item in list_data]

kwvectorizer = TfidfVectorizer(
    strip_accents='unicode',
    ngram_range=(1, 1), 
    stop_words='english', 
    min_df=0.01, 
    max_df=0.90
)
documents = obtain_value_list(collated_data, 'text_no_SW')  


documents = obtain_value_list(collated_data, 'text_no_SW') 
documents = [' '.join(_) for _ in  documents]



# Save the vectors : doc_id2titleBertEmb, doc_id2textBertEmb, doc_id2textSvdEmb
f_name='doc_id2textSvdEmb.pkl'
if os.path.exists(f_name):
    with open(f_name,'rb') as fh:
        doc_id2textSvdEmb = pickle.load(fh)
else:
    t0 =  time()
    xformed_docs = kwvectorizer.fit_transform(documents)
    t1 = time()
    print(' Time taken for TF-IDF {:.4f}'.format(t1-t0))
    t0 = time()
    svd_obj = TruncatedSVD(n_components=512)
    svd_obj.fit(xformed_docs.todense())
    with open('svd_obj.pkl','wb') as fh:
        pickle.dump(svd_obj, fh, pickle.HIGHEST_PROTOCOL)   
    t1 = time()
    print(' Time taken for SVD fit {:.4f}'.format(t1-t0))
    
    doc_id2textSvdEmb = OrderedDict({})
    for _item  in  collated_data:
        _id =_item['synID']
        _text = kwvectorizer.transform(_item['text_no_SW'])        
        doc_id2textSvdEmb[_id] = svd_obj.transform(_text.todense())[0]
        
    with open(f_name,'wb') as fh:
        pickle.dump( doc_id2textSvdEmb, fh, pickle.HIGHEST_PROTOCOL)

f_name='doc_id2BertEmb.pkl'
if os.path.exists(f_name):
    with open(f_name,'rb') as fh:
        doc_id2textBertEmb = pickle.load(fh)
else:
    documents = obtain_value_list(collated_data, 'text_w_SW')  
    id_list = obtain_value_list(collated_data,'synID')
    emb = get_sentence_embedding_sBert(documents)
    doc_id2textBertEmb = OrderedDict({_id: _emb for _id,_emb in zip(id_list, emb)})
    
    with open(f_name,'wb') as fh:
        pickle.dump( doc_id2textBertEmb,fh, pickle.HIGHEST_PROTOCOL)
    

f_name='doc_id2LongformerEmb.pkl'
if os.path.exists(f_name):
    with open(f_name,'rb') as fh:
        doc_id2textLFormerEmb = pickle.load(fh)
else:
    documents = obtain_value_list(collated_data, 'text_w_SW')  
    id_list = obtain_value_list(collated_data,'synID')
    emb  = []
    for d in documents:
        emb.append(get_sentence_embedding_Longformer([d]))
    emb = np.array(emb)
#     emb = get_sentence_embedding_Longformer(documents)
    doc_id2textLFormerEmb = OrderedDict({_id: _emb for _id,_emb in zip(id_list, emb)})
    with open(f_name,'wb') as fh:
        pickle.dump( doc_id2textLFormerEmb,fh, pickle.HIGHEST_PROTOCOL)

def set_up_index(
    docId_list,
    vectors,
    _typeID,
    refresh = False
):
    filename = 'faiss_index_{}'.format(_typeID)
    if refresh is False and os.path.exists(filename):
        index = faiss.read_index(filename)
        return index
    
    input_count = vectors.shape[0]                  
    d = vectors.shape[1]                            
    m = 16
    k = 4
    quantizer_text = faiss.IndexFlatL2(d) 
    index = faiss.IndexIVFPQ(
        quantizer_text, 
        d, 
        input_count, 
        m, 
        k
    )

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

# Create mapping 
synID = obtain_value_list(collated_data, 'synID') 
_id = obtain_value_list(collated_data, 'id') 
_path =  obtain_value_list(collated_data, 'path') 
xml_date = obtain_value_list(collated_data, 'xml_date') 
df_Mapping = pd.DataFrame({'synID':synID , 'id': _id, 'path': _path, 'xml_date': xml_date})
df_Mapping.to_csv('mapping_data.csv',index=None)

docId_list = obtain_value_list(collated_data,'synID')
vectors_text_tfidf = np.array(list(doc_id2textSvdEmb.values())).astype(np.float32)
vectors_text_sBert = np.array(list(doc_id2textBertEmb.values())).astype(np.float32)

vectors_text_LFormer = np.array(list(doc_id2textLFormerEmb.values())).astype(np.float32)

index_tfid = set_up_index(
    docId_list = docId_list,
    vectors= vectors_text_tfidf,
    _typeID ='tfidf'
)

index_sBert = set_up_index(
    docId_list = docId_list,
    vectors= vectors_text_sBert,
    _typeID ='sBert'
)

index_LFormer = set_up_index(
    docId_list = docId_list,
    vectors= vectors_text_LFormer,
    _typeID ='Longformer'
)

