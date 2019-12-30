# -*- coding: utf-8 -*-
import dill

class model():
    def __init__(self,name,path=None,model_name=""):
        self.name = name
        self.path = path
        self.model_name = model_name
        self.model_data = None
        self.load_model()
    def load_model(self):
        if self.path is not None:
            with open(self.path ,'rb') as f:
                self.model_data = dill.load(f)
    def train(self):
        if self.name == "naive_bayes":
            pass