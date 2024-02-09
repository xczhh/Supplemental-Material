import pickle,config
# from Capsule_Keras import *
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
#from keras_utils import Capsule, AttentionWithContext,Attention
from utils.Capsule import capsule
from utils.att import AttentionWithContext, AttentionLayer, Attention
import numpy
from utils.focal_loss import focal_loss
#import dataProcess
num_classes = 4


def get_hier_attLSTM(maxLen, embWeights=None,embeddingSize=None, vocabSize=None, n_recurrent=64, dropout_rate=0.2, l2_penalty=0.0001, mask_zero = True):
    '''
    GRU-based RNN
    :return: Model
    '''
    word_input = Input(shape=(maxLen,), dtype="int32")
    if embWeights:
        embed = Embedding(embWeights[0].shape[0], embWeights[0].shape[1], weights=[embWeights[0]], trainable=False)(word_input)
    else:
        embed = Embedding(vocabSize, embeddingSize)(word_input)

    x = SpatialDropout1D(dropout_rate)(embed)
    bi_x = Bidirectional(
        CuDNNGRU(n_recurrent, return_sequences=True,
                 kernel_regularizer=l2(l2_penalty),
                 recurrent_regularizer=l2(l2_penalty)))(x)
    # bi_x = Bidirectional(
    #         CuDNNGRU(n_recurrent, return_sequences=True))(x)

    att_x = Attention(maxLen)(bi_x)
    gap_x= GlobalAveragePooling1D()(bi_x)
    gmp_x= GlobalMaxPool1D()(bi_x)

    x = concatenate([att_x, gap_x, gmp_x])

    modelSentence = Model(word_input, x)

    documentInputs = Input(shape=(None, maxLen), dtype='int32', name='document_input')
    sentenceEmbbeding = TimeDistributed(modelSentence)(documentInputs)
    sentenceRnn = LSTM(150, return_sequences=True)(
        sentenceEmbbeding)
    documentEmb = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x: (x[0], x[2]), name="att2")(sentenceRnn)
    # weights 
    
    outputs = Dense(num_classes, activation="softmax")(documentEmb)
    model = Model(inputs=documentInputs, outputs=[outputs,documentEmb])
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=['accuracy'])
    print(model.summary())
    return model



