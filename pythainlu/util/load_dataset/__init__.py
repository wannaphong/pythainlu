# -*- coding: utf-8 -*-
import codecs
def load_txt(file):
    with codecs.open(file, 'r',encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    return lines