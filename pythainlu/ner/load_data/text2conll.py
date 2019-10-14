# -*- coding: utf-8 -*-
import codecs
from pythainlu.util import load_txt
from nltk.tokenize import RegexpTokenizer
from pythainlp.tag import pos_tag
import nltk
import re

# เตรียมตัวตัด tag ด้วย re
pattern = r'\[(.*?)\](.*?)\[\/(.*?)\]'
tokenizer = RegexpTokenizer(pattern) # ใช้ nltk.tokenize.RegexpTokenizer เพื่อตัด [TIME]8.00[/TIME] ให้เป็น ('TIME','ไง','TIME')
data_not=[]
# ใช้สำหรับกำกับ pos tag เพื่อใช้กับ NER
# print(text2conll2002(t,pos=False))
def postag(text):
    listtxt = [i for i in text.split('\n') if i!='']
    list_word = []
    for data in listtxt:
        list_word.append(data.split('\t')[0])
    list_word = pos_tag(list_word,engine="perceptron", corpus="orchid_ud")
    text = ""
    i = 0
    for data in listtxt:
        text += data.split('\t')[0]+'\t'+list_word[i][1]+'\t'+data.split('\t')[1]+'\n'
        i += 1
    return text

def Unique(p):
    global data_not
    text = re.sub("<[^>]*>","",p)
    text = re.sub("\[(.*?)\]","",text)
    text = re.sub("\[\/(.*?)\]","",text)
    if text not in data_not:
        data_not.append(text)
        return True
    else:
        return False
# จัดการกับ tag ที่ไม่ได้ tag
def toolner_to_tag(text):
    text=text.strip()
    text=re.sub("<[^>]*>","",text)
    text=re.sub("(\[\/(.*?)\])","\\1***",text)# ตัดการกับพวกไม่มี tag word
    text=re.sub("(\[\w+\])","***\\1",text)
    text2=[]
    for i in text.split('***'):
        if "[" in i:
            text2.append(i)
        else:
            text2.append("[word]"+i+"[/word]")
    text="".join(text2)
    return text.replace("[word][/word]","")

# แปลง text ให้เป็น conll2002
def text2conll2002(text,word_seg,pos=True,postag=None):
    """
    ใช้แปลงข้อความให้กลายเป็น conll2002
    """
    text = toolner_to_tag(text)
    text = text.replace("''",'"').replace("’",'"').replace("‘",'"')#.replace('"',"")
    tag = tokenizer.tokenize(text)
    j = 0
    conll2002 = ""
    for tagopen,text,tagclose in tag:
        word_cut = word_seg(text)
        i = 0
        txt5 = ""
        while i < len(word_cut):
            if word_cut[i] == "''" or word_cut[i] == '"':pass
            elif i == 0 and tagopen != 'word':
                txt5 += word_cut[i]
                txt5 += '\t'+'B-'+tagopen
            elif tagopen != 'word':
                txt5 += word_cut[i]
                txt5 += '\t'+'I-'+tagopen
            else:
                txt5 += word_cut[i]
                txt5 += '\t'+'O'
            txt5 += '\n'
            i += 1
        conll2002 += txt5
    if pos == False:
        return conll2002
    return postag(conll2002)

# อ่านข้อมูลจากไฟล์
def get_data(fileopen):
	return [a for a in load_txt(fileopen) if Unique(a)]

def alldata_list(lists,word_seg,pos=True,fn_postag=None):
    data_all = []
    if fn_postag == None and pos:
        fn_postag=postag
    for data in lists:
        data_num = []
        try:
            txt = text2conll2002(data,word_seg=word_seg,pos=pos,postag=fn_postag).split('\n')
            for d in txt:
                tt = d.split('\t')
                if d != "":
                    if len(tt) == 3:
                        data_num.append((tt[0],tt[1],tt[2]))
                    else:
                        data_num.append((tt[0],tt[1]))
            data_all.append(data_num)
        except Exception as e:
            print(e)
    return data_all

def alldata_list_str(lists):
	string = ""
	for data in lists:
		string1 = ""
		for j in data:
			string1 += j[0]+"	"+j[1]+"	"+j[2]+"\n"
		string1 += "\n"
		string += string1
	return string

def get_data_tag(listd):
	list_all = []
	c = []
	for i in listd:
		if i != '':
			c.append((i.split("\t")[0],i.split("\t")[1],i.split("\t")[2]))
		else:
			list_all.append(c)
			c = []
	return list_all

def getall(lista):
    ll = []
    for i in lista:
        o = True
        for j in ll:
            if re.sub("\[(.*?)\]","",i)==re.sub("\[(.*?)\]","",j):
                o = False
                break
        if o == True:
            ll.append(i)
    return ll