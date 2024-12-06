from tensorflow.keras import layers
import csv
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, MaxPooling1D
from sklearn import preprocessing
from sklearn.metrics import r2_score
from tensorflow.keras.losses import mean_squared_error
from KS import ks
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import datetime
now = datetime.datetime.now()

times = now.strftime("%m%d%H%M")
csv_logger = CSVLogger('./out/train_all/'+times +
                       'training_log.csv', separator=',', append=True)
batch_size, timesteps, input_dim = None, 201, 1


def root_mean_squared_error(y_true, y_pred):
    loss = K.sqrt(mean_squared_error(y_true, y_pred))
    return loss


def my_init(shape, dtype=None):
    return 0.1*K.random_normal(shape, dtype=dtype)


class MyCallback(Callback):
    def __init__(self):
        self.counter = 0
        self.counter2 = 0

    def on_epoch_end(self, epoch, logs=None):
        r2 = logs.get('r2_train')

        loss = logs.get('loss')
        if np.isnan(loss):
            with open('nan.csv', 'a') as f:
                    f.write(f'{loss}\n')
            return


def DcnnModel(X_train, X_vaild, y_train, y_vaild, sd):

    inputs = Input(batch_shape=(batch_size, timesteps, input_dim))

    # filters1     =  8
    # filters2     =  6
    # kernel_size1 =  9
    # kernel_size2 =  13
    # lr           = 0.002
    # Dropout1     = 0.1
    filters1 = 8
    filters2 = 8
    kernel_size1 = 3
    kernel_size2 = 3
    lr = 0.01
    Dropout1 = 0.3
    # filters1     =  machine_assignment[0] 8
    # filters2     =  machine_assignment[1] 6
    # kernel_size1 =  machine_assignment[2] 9
    # kernel_size2 =  machine_assignment[3] 13

    # filters3     =  machine_assignment[4]
    # kernel_size4 =  machine_assignment[5]

    # lr           = round(machine_assignment[6], 4) 0.002
    # Dropout1     = round(machine_assignment[7], 1) 0.1
    # 训练个数  步长  数据的维度

    dcnn = Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', dilation_rate=2,
              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4))(inputs)  # Elastic Net正则化
    dcnn = MaxPooling1D(2, strides=2)(dcnn)
    dcnn = Conv1D(filters=filters2, kernel_size=kernel_size1, activation='relu', dilation_rate=4,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4))(dcnn)  # Elastic Net正则化

    # Squeeze-and-Excitation
    squeeze = layers.GlobalAveragePooling1D()(dcnn)
    excitation = layers.Dense(units=filters2 // 4, activation='relu')(squeeze)
    excitation = layers.Dense(units=filters2, activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, filters2))(excitation)
    dcnn_with_attention = layers.multiply([dcnn, excitation])

    dcnn_with_attention = Flatten()(dcnn_with_attention)
    dcnn_with_attention = Dense(200, activation='relu')(dcnn_with_attention)
    dcnn_with_attention = Dropout(Dropout1)(dcnn_with_attention)
    dcnn_with_attention = Dense(100, activation='relu')(dcnn_with_attention)
    dcnn_with_attention = Dropout(Dropout1)(dcnn_with_attention)
    dcnn_with_attention = Dense(1, activation='relu')(dcnn_with_attention)

    dcnn_with_attention = Model(inputs=[inputs], outputs=[dcnn_with_attention])

    dcnn = dcnn_with_attention
    print(dcnn_with_attention.summary())
    def r2_train(y, y_pred):
        ss_res = K.sum(K.square(y - y_pred))  # 残差平方和
        ss_tot = K.sum(K.square(y - K.mean(y)))  # 总平方和
        r2 = 1 - ss_res / ss_tot  # 计算R²
        return r2
 
    

    
    adam = tf.keras.optimizers.Adam(lr=lr)
    my_callback = MyCallback()

    dcnn.compile(optimizer=adam, loss=root_mean_squared_error, metrics=[r2_train])#, metrics=[r2_train]

    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
    #monitor参数用于指定要监视的指标，patience参数用于指定在停止训练之前要等待多少个epoch。
    # monitor：要监测的指标。
    # patience：当early stopping被激活（如发现loss没有下降），则经过patience个epoch后停止训练。
    # verbose：信息展示模式。0表示不展示，1表示展示。
    # mode：‘auto’，‘min’，‘max’之一。如果是min模式，则当监测指标停止下降时停止训练；如果是max模式，则当监测指标停止上升时停止训练；如果是auto，则根据被监测的数据自动选择模式。
    # restore_best_weights：当设置为True时，将在停止前保存最佳模型权重。
    history = dcnn.fit(X_train, y_train, epochs=2000, batch_size=64,
                      validation_data=(X_vaild, y_vaild),callbacks=[my_callback,early_stopping, csv_logger])
                      #validation_data=(X_vaild, y_vaild),callbacks=[my_callback,early_stopping])


    rmse__train = history.history['loss'][-1]
    r2__train = history.history['r2_train'][-1]
    rmse__validation = history.history['val_loss'][-1]
    r2__validation = history.history['val_r2_train'][-1]
    
    rmse__train *= 100
    rmse__validation *= 100

    print(  "---------------------------------"
            "filters:",filters1,'\n',filters2,'\n',
            "kernel_size:",kernel_size1,'\n',kernel_size2,'\n',
            "drop:",lr,'\n',
            "lr:",Dropout1,'\n',
            "rmse_train:",  rmse__train,
            "r^2_train:", r2__train,
            "rmse_validation:", rmse__validation,
            "r^2_validation:", r2__validation,
            "---------------------------------"
            )

    history.history.keys()

    # plt.figure()#画在一张图上
    # plt.plot(history.epoch, history.history.get('loss'))
    # plt.plot(history.epoch, history.history.get('val_loss'))
    # plt.plot(history.epoch, history.history.get('loss'))
    # plt.plot(history.epoch, history.history.get('val_loss'))
    # plt.xlim((20, 10000))
    # plt.ylim((0, 1))

    y_train_pre = dcnn.predict(X_train)
    y_vaild_pre = dcnn.predict(X_vaild)


    # print('预测！！！！！！！！！！！！！！！！！！')
    
    
    # def rmse( y_pred, y_true):
    #     return np.sqrt(np.mean((y_true - y_pred) ** 2))

    # def r2( y_pred, y_true):
    #     return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # rmse_train = rmse(y_train_pre, y_train)
    # rmse_vaild = rmse(y_vaild_pre, y_vaild)
    # r2_train = r2(y_train_pre, y_train)
    # r2_vaild = r2(y_vaild_pre, y_vaild)
    # fitness = (rmse_train + rmse_vaild) / 2
    # print(fitness, rmse_train, rmse_vaild, r2_train, r2_vaild)
    # print('预测！！！！！！！！！！！！！！！！！！')   

    return y_train_pre, y_vaild_pre