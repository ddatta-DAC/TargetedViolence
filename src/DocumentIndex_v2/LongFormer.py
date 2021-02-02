#!/usr/bin/env python
# coding: utf-8

# In[6]:


from transformers import AutoTokenizer, AutoModel
import torch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

DEVICE = torch.device('cpu')    
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModel.from_pretrained("allenai/longformer-base-4096").to(DEVICE)

# =============================
#  Input should be a list of sentences.
# =============================
def get_doc_emb(doc_text):
    global tokenizer
    global model
    global DEVICE
    
    encoded_input = tokenizer(doc_text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Take a mean of the sentences 
    return torch.mean(sentence_embeddings,dim=-2).cpu().data.numpy()


# #Sentences we want sentence embeddings for
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']


# get_doc_emb(sentences)

