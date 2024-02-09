# -*- coding: utf-8 -*-
import numpy
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.callbacks.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout,multiply
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.engine.topology import Layer
from keras import regularizers
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn import model_selection, naive_bayes, svm, tree
from keras.layers.wrappers import Bidirectional
from model import get_hier_attLSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, load_vocabulary
import numpy as np
from keras_bert import extract_embeddings
#from imblearn.datasets import fetch_datasets
#from data.over_sampling.kmeans_smote.kmeans_smote import KMeansSMOTE
np.random.seed(1)
from utils import sen, spe, encodeLabel
import time
start_time = time.time()

maxLen = 150
# word2vec的训练:
# 设置词语向量维度
num_featrues = 100
# 保证被考虑词语的最低频度
min_word_count = 3
# 设置并行化训练使用CPU计算核心数量
num_workers = 4
# 设置词语上下文窗口大小
context = 4
vec_path = "E:\\compare_test\\data\\over_sampling\\words300"
config_path = "data/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "data/chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "data/chinese_L-12_H-768_A-12/vocab.txt"

# texts = ['all work and no play', 'makes jack a dull boy~']
#embeddings = extract_embeddings("data/chinese_L-12_H-768_A-12", texts)
#bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=False)



wordCut = []
train_x = []
train_y = []
test_x = []
test_y = []
train = []
test = []

# 1. load train data
with open("train.txt", encoding="utf-8") as f:
     train = f.readlines()
for i in range(0, len(train)):
    content = train[i]
    content = content.strip().split('\t')
    train_x.append(content[0])
    print(content[0])
    train_y.append(content[1])

# 2. load test data
with open("test.txt", encoding="utf-8") as f:
    test = f.readlines()
for i in range(0, len(test)-1):
    content = test[i]
    content = content.strip().split('\t')
    test_x.append(content[0])
    test_y.append(content[1])

# 3. initialize word embeddings
word_embeddings = []
word2id = {}
f = open(vec_path, "r", encoding='utf-8')
content = f.readline()
while True:
    content = f.readline()
    if content == "":
        break
    content = content.strip().split()
    word2id[content[0]] = len(word2id)  # word->id
    content = content[1:]
    content = [(float)(i) for i in content]
    word_embeddings.append(content)
f.close()
word2id['UNK'] = len(word2id)  # word2id[28] word_embeddings[26, 100]
word2id['BLANK'] = len(word2id)
lists = [0.0 for i in range(len(word_embeddings[0]))]  # lists: [100]
word_embeddings.append(lists)
word_embeddings.append(lists)
word_embeddings=np.array(word_embeddings)

# 4. word vectors
max_sequence = 150
size = len(train_x)
length_train = []
length_test = []
blank = word2id['BLANK']
for i in range(size):
    text = [blank for j in range(max_sequence)]
    content = train_x[i].split()
    for j in range(len(content)):
        if (j == max_sequence):
            break
        if not content[j] in word2id:
            text[j] = word2id['UNK']
        else:
            text[j] = word2id[content[j]]
    train_x[i] = text

train_x = np.array(train_x)
token_dict = load_vocabulary(dict_path)  # {word:id,word1:id1}

size = len(test_x)
for i in range(size):
    text = [blank for j in range(max_sequence)]
    content = test_x[i].split()
    for j in range(len(content)):
        if (j == max_sequence):
            break
        if not content[j] in word2id:
            text[j] = word2id['UNK']
        else:
            text[j] = word2id[content[j]]
    test_x[i] = text

trainLable = encodeLabel(train_y)
train_y = to_categorical(trainLable, num_classes=4)  # 将标签转换为one-hot编码

# 5. initialize model
model,modelev = get_hier_attLSTM(maxLen, embWeights=word_embeddings, embeddingSize=200, vocabSize=20000)#embWeights=bert_model)#embWeights=word_embeddings, embeddingSize=200, vocabSize=20000)


train_x = np.expand_dims(train_x,axis=1)
test_x = np.expand_dims(test_x,axis=1)

trainLable = encodeLabel(test_y)
test_y = to_categorical(trainLable, num_classes=4)

class PredictionCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
      modelev.save('modelvisual/'+str(epoch)+'modelev.h5') # weights visualization
      pred = model.predict(test_x, batch_size=116, verbose=1)
      loss, acc = model.evaluate(test_x, test_y)
      # val_loss['batch'].append(logs.get('val_loss'))
      print("---loss:" + str(loss) + ",acc:" + str(acc) + "----")
      predicted = np.argmax(pred, axis=1)
      real_y = np.argmax(test_y, axis=1)
      report = classification_report(real_y, predicted)
      print("sensitive ->", sen(real_y, predicted, 4))
      print("specificity ->", spe(real_y, predicted, 4))
      print(report)

model.fit(train_x, train_y, batch_size=16, epochs=50, validation_data=(test_x, test_y),
           callbacks=[PredictionCallback()])

end_time = time.time()
run_time = end_time - start_time
print("代码运行时间为：%f秒" % run_time)
# model.save('model_h/model.h5')

