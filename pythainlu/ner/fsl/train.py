from pythainlu.ner.fsl.model import CNN_BLSTM
from pythainlp.tokenize import word_tokenize
from pythainlu.ner.load_data import getall,get_data,alldata_list
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Nadam

"""Set parameters"""

#EPOCHS = 30               # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 200     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0105    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended

"""Construct and run model"""


def train(
    name:str,
    path_data:str,
    path:str="./",
    test:bool=False,
    test_size:float=0.2,
    word_seg=word_tokenize,
    ep=30):
    cnn_blstm = CNN_BLSTM(ep, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)
    data = getall(get_data(path_data))
    datatofile = alldata_list(data,word_seg=word_seg)
    data_train,data_test= train_test_split(datatofile, test_size=test_size)
    cnn_blstm.loadData(data_train,data_test,data_train)
    cnn_blstm.addCharInfo()
    cnn_blstm.embed()
    cnn_blstm.createBatches()
    cnn_blstm.buildModel()
    cnn_blstm.train()
    cnn_blstm.writeToFile()