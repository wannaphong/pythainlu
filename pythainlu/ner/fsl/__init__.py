# -*- coding: utf-8 -*-
"""
Keras implementation of "Few-shot Learning for Named Entity Recognition in Medical Text"

Code from https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL
"""
from pythainlu.ner.fsl.model import CNN_BLSTM

class model:
    def __init__(self,path_model,word_seg=word_tokenize):
        """
        CRF named-entity recognizer

        :param string path_model: path model
        :param function features: features function
        :param function word_seg: word_tokenize function
        """
        self.__data_path = path_model
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=500,
            all_possible_transitions=True,
            model_filename=self.__data_path,
        )
        self.features=features
        self.word_seg=word_seg

    def get_ner(self, text: str, pos: bool = True, tag: bool = False):
        pass
