# -*- coding: utf-8 -*-
"""
Code from https://www.kaggle.com/akshay235/bert-implementation-on-ner-corpus
"""
from pythainlp.tokenize import word_tokenize
from pythainlu.ner.load_data import getall,get_data,alldata_list
import numpy as np
from .data_load import NerDataset
from .b_train import train_go

def train(name:str,path_data:str,path:str="./",test:bool=False,test_size:float=0.2,word_seg=word_tokenize,ep=30,lr=0.0001,batch_size=128,finetuning=False,top_rnns=False):
    data = getall(get_data(path_data))
    datatofile = alldata_list(data,word_seg=word_seg,pos=False)
    tag_all=['<PAD>']
    for sent in datatofile:
        tag_all.extend([line[-1] for line in sent])
        #print(sent)
    tag_all = list(set(tag_all))
    #print(tag_all)
    if not test:
        train_dataset = datatofile
        train_go(train_dataset,tag_all,path,ep=ep,lr=lr,batch_size=batch_size,finetuning=finetuning,top_rnns=top_rnns)