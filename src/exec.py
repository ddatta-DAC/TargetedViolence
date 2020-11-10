import os
import glob
import sys
import yaml
import numpy as np
import pandas as pd
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


REFRESH = False
RAW_ID_KEY = 'id'
json_data_key_1 = 'SpacyEnrichment'
json_data_key_2 = 'tokens'
json_data_key_3 = 'lemma'
MIN_WORD_LEN = 3
saved_data_loc = 'saved_model_data'
pathobj = Path(saved_data_loc)
pathobj.mkdir(parents=True, exist_ok=True)
CONFIG = None
with open(r'config.yaml') as fh:
    CONFIG = yaml.load(fh, Loader=yaml.SafeLoader)
print(CONFIG)

files = sorted(glob.glob(os.path.join(CONFIG['DATA_LOC'], CONFIG['data_file_pattern'])))

file = files[0]
file 

def aux_process_file(file):
    global RAW_ID_KEY
    file_data = {}
    with open(file,'rb') as fh:
        for line in fh:    
            _dict = json.loads(line)
            file_data[_dict[RAW_ID_KEY]] = _dict
            
    return aux_process_fileData(file_data)       


# d1 = aux_process_file(file)
# list(d1.keys())[0]

def aux_process_fileData(file_data):
    global MIN_WORD_LEN
    global json_data_key_3
    docID_words_dict = {}
    for docID, doc in file_data.items():
        words = []
        for token in doc [json_data_key_1][json_data_key_2]:
            if len(token[json_data_key_3]) > MIN_WORD_LEN:  words.append(token['lemma'])
        docID_words_dict[docID] = words
    return docID_words_dict

DOC_DICT_FILE = os.path.join(saved_data_loc,'documents_dict.pkl')
if REFRESH or not os.path.exists(DOC_DICT_FILE):

    list_docs = Parallel(mp.cpu_count())(delayed(aux_process_file)(file) for file in files)
    # ----------------------
    # Collate 
    # ----------------------
    docID_words_dict = {}
    for _ in list_docs:
        docID_words_dict.update(_)

    docID_words_dict = OrderedDict(docID_words_dict)
    with open(DOC_DICT_FILE,'wb') as fh:
        pickle.dump(docID_words_dict, fh, pickle.HIGHEST_PROTOCOL)
else:
    with open(DOC_DICT_FILE,'wb') as fh:
        docID_words_dict = pickle.load(fh)
    
    

#  ----------------------------------------------------
# Create synthetic IDs
# -----------------------------------------------------
docID_list = list(docID_words_dict.keys())
docID_to_synID = {e[1]:e[0] for e in enumerate(docID_list,0)}
docID_synID_df = pd.DataFrame( {'docID':docID_list, 'synID': np.arange(len(docID_list))}, columns=['docID','synID'])

# ------------------------------------------------------
# Create TF IDF vectorizer
# ------------------------------------------------------
t0 = time()
kwvectorizer = TfidfVectorizer(
    strip_accents='unicode',
    ngram_range=(1, 1), 
    stop_words='english', 
    min_df=0.01, 
    max_df=0.80
)
documents = list(docID_words_dict.values())



xformed_docs = kwvectorizer.fit_transform([ ' '.join(_) for  _ in documents])
t1 = time()
print(' Time taken for TF-IDF {:.4f}'.format(t1-t0))
t0 = time()
svd_obj = TruncatedSVD(n_components=128)
xformed_docs_SVD = svd_obj.fit_transform(xformed_docs.todense())
t1 = time()
print(' Time taken for SVD {:.4f}'.format(t1-t0))
xformed_docs_SVD = xformed_docs_SVD.astype(np.float32)
xformed_docs_SVD.shape, type(xformed_docs_SVD)

vectors = xformed_docs_SVD
input_count = vectors.shape[0] #Input count
d = vectors.shape[1] # Input dimension 
m = 16
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same 

index = faiss.IndexIVFPQ(
    quantizer, 
    d, 
    input_count, 
    m, 
    k
)
arr_ids = docID_synID_df['synID'].values.astype(np.int)

t0 = time()
index.train(vectors)
t1 = time()
print('Is Index Trained ?', index.is_trained)

index.add_with_ids(
    vectors, 
    arr_ids 
)
t2 = time()
print('Time taken for Train {:.5f} ||  Adding index {:.5f}'.format(t1-t0, t2-t1))






def benchmark_time(index, vectors, docs_count = 1000, find_nn=10):

    index.nprobe = 10
    idx = np.arange(vectors.shape[0])
    np.random.shuffle(idx)
    idx = idx[:docs_count]
    t0 = time()
    D, I = index.search(vectors[idx], find_nn) # sanity check
    t1 = time()
    print('Time taken {:.4f}'.format(t1-t0))
    return (t1-t0)


benchmark_time(index, vectors)

benchmark_time(index, vectors)

benchmark_time(index, vectors)
