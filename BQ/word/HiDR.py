# -*- coding: utf-8 -*-
import nni
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.backend import stack, squeeze, flatten, reshape
from keras.layers import Concatenate, AveragePooling2D
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
import numpy as np
from keras import backend as K, backend
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Input, Bidirectional, Lambda, LSTM, \
    Reshape, Dense, Activation, concatenate, BatchNormalization, Conv2D, GRU, \
    MaxPooling2D, Flatten
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, Adadelta, RMSprop, SGD, Adagrad, Adamax, Nadam
from keras.preprocessing.sequence import pad_sequences
import data_helper

drop = 0.6
input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    return precision


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    recall = c1 / c3
    return recall



embedding_matrix = data_helper.load_pickle('./embedding_matrix.pkl')
embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)


def base_network(input_shape,params):
    input_ = Input(shape=input_shape)

    embed = embedding_layer(input_)
    
    p_list = []
    for i in range(2):
        p_ = Bidirectional(LSTM(params['hid_dim'],
                            return_sequences=True,
                            dropout=drop),
                       merge_mode='sum')(embed)
        p = Lambda(lambda x: K.sum(x, axis=1), output_shape=(params['hid_dim'],))(p_)
        p_list.append(p)

    ad = Lambda(lambda x: stack(x, axis=1))(p_list)
    asd = Reshape((3, params['hid_dim'], 1))(ad)

    con1 = Conv2D(filters=params['conv_filters'], kernel_size=(2, params['conv12_kenel']), padding='valid', strides=[1, params['conv12_strides']],
                  data_format='channels_last',
                  activation='relu')(asd)

    con2 = Conv2D(filters=params['conv_filters'], kernel_size=(2, params['conv12_kenel']), padding='valid', strides=[1, params['conv12_strides']],
                  data_format='channels_last',
                  activation='relu')(asd)

    pool_con = Concatenate(axis=1)([con1, con2])

    con3 = Conv2D(filters=params['conv_filters'], kernel_size=(2, params['conv3_kenel']), padding='Valid', strides=[1, params['conv3_strides']], data_format='channels_last',
                  activation='relu')(pool_con)

    model = Model(input_, con3, name='review_base_nn')

    return model

def siamese_model(params):
    input_shape = (input_dim,)

    base_net = base_network(input_shape,params)

    input_q1 = Input(shape=input_shape, dtype='int32', name='sequence1')

    processed_q1 = base_net([input_q1])

    input_q2 = Input(shape=input_shape, dtype='int32', name='sequence2')

    processed_q2 = base_net([input_q2])

    pros = Concatenate(axis=1)([processed_q1, processed_q2])
    pro_filters = 128
    pros_conv = Conv2D(filters=pro_filters, kernel_size=(2,2), padding='Valid', strides=[1, 1], data_format='channels_last',
           activation='relu')(pros)

    _dim1 = backend.int_shape(pros_conv)[1]
    _dim2 = backend.int_shape(pros_conv)[2]

    similarity = Reshape((_dim1*_dim2*pro_filters,))(pros_conv)

    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)

    model = Model([input_q1, input_q2], [similarity])
    op = RMSprop(lr=params['learning_rate'])

    model.compile(loss="binary_crossentropy", optimizer=op, metrics=['accuracy', precision, recall, f1_score])
    return model

def train(params):
    data = data_helper.load_pickle('./model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_y = data['dev_label']

    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_y = data['test_label']

    model = siamese_model(params)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max',restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, mode='max')
    callbackslist = [checkpoint, tensorboard, earlystopping, reduce_lr]


    model.fit([train_q1, train_q2], train_y,
              batch_size=512,
              epochs=200,
              verbose=2,
              validation_data=([dev_q1, dev_q2], dev_y),
              callbacks=callbackslist)

    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2], test_y, verbose=1, batch_size=256)
    print("Test best model =loss: %.4f, accuracy:%.4f,precision:%.4f,recall:%.4f,f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score))
    # nni.report_final_result(accuracy)


if __name__ == '__main__':
    # RECEIVED_PARAMS = nni.get_next_parameter()

    params = {"learning_rate":0.0011509815585850523,"hid_dim":400,"conv_filters":64,"conv12_kenel":2,"conv12_strides":1,"conv3_kenel":2,"conv3_strides":1,"optimizer":"RMSprop","loss":"msle"}  #
    # params.update(RECEIVED_PARAMS1)
    print("biaozhun_BQ:")
    train(params)
