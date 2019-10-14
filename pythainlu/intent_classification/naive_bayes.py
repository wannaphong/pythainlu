# -*- coding: utf-8 -*-
from typing import List
import nltk
def train(train_data:List[tuple],get_features:object,test_data=None)->tuple:
    """
    Train Naive Bayes Classification

    :param list train_data: list train data of tuple (text,tag)
    :param object get_features: function of features
    :param list test_data: list test data of tuple (text,tag)

    :return: tuple(model,accuracy)
    """
    data_train = [(get_features(text), tag) for (text, tag) in train_data]
    classifier = nltk.NaiveBayesClassifier.train(data_train)
    if test_data!=None:
        data_test= [(get_features(text), tag) for (text, tag) in test_data]
        return (classifier, nltk.classify.accuracy(classifier, data_test))
    return (classifier,)
def predict(model,text,get_features):
    feature = get_features(text)
    return (model.classify(feature), model.prob_classify(feature))