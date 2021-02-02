#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import gensim
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import collections


def get_taggedDocCorpus(list_docs, list_ids):
    return TaggedDocument(list_docs, list_ids)


def build_model (list_ids, list_docs, epochs = 100):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=256, hs = 1, min_count= 3, epochs=100)
    train_corpus = get_taggedDocCorpus(list_docs, list_ids)
    documents = [TaggedDocument(doc, [i]) for doc, i in zip(list_docs, list_ids,)]
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model.docvecs, model

def get_docVector(modelObj, doc):
    vector = modelObj.infer_vector(doc)
    return vector


# In[36]:


# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)

#     second_ranks.append(sims[1])
# counter = collections.Counter(ranks)
# print(counter)


# In[ ]:




