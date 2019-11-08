# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pythainlp import word_tokenize
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback
from pythainlp.ulmfit import *

tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)

processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=None, max_vocab=60000, min_freq=2)]

def train(name,train_data:List[tuple],num_tag,test_data=None,model_path,="./"):
    data_train = pd.DataFrame(train_data)
    all_df=data_train
    if test_data != None:
        data_test = pd.DataFrame(test_data)
        all_df+=data_test
    # LM
    data_lm = (TextList.from_df(all_df, model_path, cols="texts", processor=processor)
    .split_by_rand_pct(valid_pct = 0.01, seed = 1412)
    .label_for_lm()
    .databunch(bs=48)
    data_lm.sanity_check()
    data_lm.save(str(name)+'_lm.pkl')
    data_lm.sanity_check()
    config = dict(emb_sz=400, n_hid=1550, n_layers=num_tag, pad_token=1, qrnn=False, tie_weights=True, out_bias=True,
             output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)
    trn_args = dict(drop_mult=1., clip=0.12, alpha=2, beta=1)
    learn = language_model_learner(data_lm, AWD_LSTM, config=config, pretrained=False, **trn_args)
    #load pretrained models
    learn.load_pretrained(**_THWIKI_LSTM)
    #train frozen
    print("training frozen")
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
    #train unfrozen
    print("training unfrozen")
    learn.unfreeze()
    learn.fit_one_cycle(5, 1e-3, moms=(0.8, 0.7))
    learn.save_encoder(str(name)+"_enc")
    data_lm = load_data(model_path, str(name)+'_lm.pkl')
    data_lm.sanity_check()