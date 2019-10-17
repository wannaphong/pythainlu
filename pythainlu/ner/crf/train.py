# -*- coding: utf-8 -*-
from pythainlp.tokenize import word_tokenize
import nltk
import re
from sklearn_crfsuite import scorers,metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import isthai
from pythainlu.ner.load_data import getall,get_data,alldata_list
stopwords = list(thai_stopwords())

def isThaiWord(word):
    return isthai(word)

def is_stopword(word):
    return word in stopwords
def is_s(word):
    if word == " " or word =="\t" or word=="":
        return True
    else:
        return False

def lennum(word,num):
    if len(word) == num:
        return True
    return False

def doc2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    # Features from current word
    features = {
        'word.word': word,
        'word.stopword': is_stopword(word),
        'word.isthai': isThaiWord(word),
        'word.isspace': word.isspace(),
        'postag': postag,
        'word.isdigit()': word.isdigit()
    }
    if word.isdigit() and len(word) == 5:
        features['word.islen5'] = True
    if i > 0:
        prevword = doc[i-1][0]
        postag1 = doc[i-1][1]
        features['word.prevword'] = prevword
        features['word.previsspace'] = prevword.isspace()
        features['word.previsthai'] = isThaiWord(prevword)
        features['word.prevstopword'] = is_stopword(prevword)
        features['word.prepostag'] = postag1
        features['word.prevwordisdigit'] = prevword.isdigit()
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    # Features from next word
    if i < len(doc)-1:
        nextword = doc[i+1][0]
        postag1 = doc[i+1][1]
        features['word.nextword'] = nextword
        features['word.nextisspace'] = nextword.isspace()
        features['word.nextpostag'] = postag1
        features['word.nextisthai'] = isThaiWord(nextword)
        features['word.nextstopword'] = is_stopword(nextword)
        features['word.nextwordisdigit'] = nextword.isdigit()
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    return features

def extract_features(doc,features_train):
    return [features_train(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [tag for (token,postag,tag) in doc]

def train(
    name:str,
    path_data:str,
    path:str="./",
    test:bool=False,
    test_size:float=0.2,
    word_seg=word_tokenize,
    features=doc2features):
    data = getall(get_data(path_data))
    datatofile = alldata_list(data,word_seg=word_seg)
    X_data = [extract_features(doc,doc2features) for doc in datatofile]
    y_data = [get_labels(doc) for doc in datatofile]
    crf = sklearn_crfsuite.CRF(
    algorithm = 'lbfgs',
    c1 = 0.1,
    c2 = 0.1,
    max_iterations = 500,
    all_possible_transitions = True,
    model_filename = path+name+".model"
    )
    if test:
        X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=test_size)
        crf.fit(X, y);
        labels = list(crf.classes_)
        labels.remove('O')
        y_pred = crf.predict(X_test)
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        
        )
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))
    else:
        crf.fit(X_data, y_data);
    return True