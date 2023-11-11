#3DDFCN///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from tensorflow._api.v2.compat.v1.initializers import zeros
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import cv2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Softmax, BatchNormalization, Activation, Dropout, concatenate, Conv2DTranspose, Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose
from keras.models import Model, load_model
#from keras.objectives import categorical_crossentropy
#from keras.utils import plot_model
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model

#import segmentation_models as sm

import math

import glob
import re

import os
#import pydicom
#from pydicom.tag import Tag
#import cv2
from IPython.display import Image, display

# 学習用のデータを作る.
image_list = []
image_list2 = []
label_list = []
label_list2 = []
img_3D = []
img_3D_2 = []
label_3D = []
label_3D_2 = []

name = 'mse_5000_shuffle_batch1'

# 2値クロスエントロピー
bce = tf.keras.losses.BinaryCrossentropy()
# 多クラスクロスエントロピー
#cce = tf.keras.losses.CategoricalCrossentropy()


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    smooth = 1. # ゼロ除算回避のための定数
    y_true_flat = tf.reshape(y_true, [-1]) # 1次元に変換
    y_pred_flat = tf.reshape(y_pred, [-1]) # 同様

    tp = tf.reduce_sum(y_true_flat * y_pred_flat) # True Positive
    nominator = 2 * tp + smooth # 分子
    denominator = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth # 分母
    score = nominator / denominator
    return 1. - score

def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 0.5 *(bce(y_true, y_pred) + dice_loss(y_true, y_pred))

def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    alpha = 0.3 # FP、FNの重み
    smooth = 1. # ゼロ除算回避のための定数
    y_true_flat = tf.reshape(y_true, [-1]) # 1次元に変換
    y_pred_flat = tf.reshape(y_pred, [-1]) # 同様

    tp = tf.reduce_sum(y_true_flat * y_pred_flat) # True Positive
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat) # False Positive
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat)) # False Negative

    score = (tp + smooth)/(tp + alpha * fp + (1-alpha) * fn + smooth) # Tversky
    return 1. - score

def focal_tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    gamma = 0.75 # 1/(4/3)=3/4=0.75
    tversky = tversky_loss(y_true, y_pred)
    return tf.pow(tversky, gamma)

#def cce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
#    return 0.5* (cce(y_true, y_pred) + dice_loss(y_true, y_pred))

'''
def dicom_process(file):
    img = pydicom.filereader.dcmread(file)
    img = img.pixel_array
    ds = pydicom.dcmread(file)
    num0 = ds.PixelSpacing
    #print(num0)

    #正規化
  #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img - 0) / (np.max(img) - 0)
        
    img = img.astype('float32')
    
    dcmshape = img.shape
    h,w = dcmshape

    cx=(num0[0]*h)/(1.0*256)
    cy=cx
    x1=h/2 
    y1=h/2 

  #画像の拡大作業開始
    x2=x1*cx #pixel
    y2=y1*cy #pixel
    size_after=(int(w*cx), int(h*cy))
    resized_img=cv2.resize(img, dsize=size_after)
    deltax=(w/2-x1)-(resized_img.shape[1]/2-x2)
    deltay=(h/2-y1)-(resized_img.shape[0]/2-y2)

    framey=int(h*cy*2)
    framex=int(w*cx*2)
    finalimg=np.zeros((framey,framex),np.float32)#########
    finalimg[int(-deltay+framey/2-resized_img.shape[0]/2):int(-deltay+framey/2+resized_img.shape[0]/2),
              int(-deltax+framex/2-resized_img.shape[1]/2):int(-deltax+framex/2+resized_img.shape[1]/2)]=resized_img
    finalimg=finalimg[int(finalimg.shape[0]/2-h/2):int(finalimg.shape[0]/2+h/2),int(finalimg.shape[1]/2-w/2):int(finalimg.shape[1]/2+w/2)]

    fimg = cv2.resize(finalimg, dsize=(256,256))

    return fimg
'''

def plot_history(history,epochs):
    #print(history.history.keys())

    
    #  損失の経過をプロット
    plt.figure()
    loss = history.history['loss']
    #val_loss = history.history['val_loss']
    plt.plot( range(epochs), loss, marker='.', label='loss' )
    #plt.plot( range(epochs), val_loss, marker='.', label='val_loss' )
    plt.legend( loc='best', fontsize=17 )
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('DFCN-model')
    plt.legend( ['loss'], loc='lower right')
    plt.savefig('result3D/graph/model_loss_3D'+name+'.png')
    plt.show()
 
    #方対数グラフ
    plt.figure()
    #loss = history.history['loss']
    #log_loss = np.log(loss)
    #val_loss = history.history['val_loss']
    plt.plot( range(epochs), loss, marker='.', label='loss' )
    #plt.plot( range(epochs), val_loss, marker='.', label='val_loss' )
    ax = plt.gca()
    ax.set_yscale('log')
    plt.legend( loc='best', fontsize=17 )
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('DFCN-model(single-logarithmic graph)')
    plt.legend( ['loss'], loc='lower right')
    plt.savefig('result3D/graph/model_loss(single-logarithmic graph)_3D'+name+'.png')
    plt.show()

    #両対数グラフ
    plt.figure()
    #loss = history.history['loss']
    #log_loss = np.log(loss)
    #val_loss = history.history['val_loss']
    plt.plot(   range(epochs), loss, marker='.', label='loss' )
    ax = plt.gca()
    ax.set_yscale('log')  # y軸をlogスケールで描く
    ax.set_xscale('log')  # x軸をlogスケールで描く
    #plt.plot( range(epochs), val_loss, marker='.', label='val_loss' )
    plt.legend( loc='best', fontsize=17 )
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('DFCN-model loss')
    plt.legend( ['loss'], loc='lower right')
    plt.savefig('result3D/graph/model_loss(double-logarithmic graph)_3D'+name+'.png')
    plt.show()




def DFCN( X_train, Xx_train, Y_train, Yy_train):
    conv_params = {'padding': 'same', 
                     'use_bias': True,#False->True
                                   'kernel_initializer': 'random_uniform',
                                   'bias_initializer': 'zeros',
                   'kernel_regularizer': None,
                   'bias_regularizer': None}
    deconv_params = {'padding': 'same',
                    'use_bias': False,
                    'kernel_initializer': 'random_uniform',
                                  'bias_initializer': 'zeros',
                    'kernel_regularizer': None,
                    'bias_regularizer': None}

    #  入力層
    #keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    #keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    #x = MaxPooling2D(pool_size= 2, strides= 2 , padding='same', data_format=None)(x)
    #cov2d stride=1///////////////////////////////////////////////////////////////////////////////////////////////////


    # CT
    input_img = Input( shape=( 128, 128, 48, 1) , name="inputs1_name")  #  32×32、RGB
    #enc0
    x = Conv3D( 32 , 3, strides= 1 , activation = 'relu' ,padding='same' )(input_img)
    f1_ct = x
    #enc1
    x = Conv3D( 64 , 3, strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(x)#半分
    f2_ct = x
    #enc2
    x = Conv3D( 128 , 3, strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(x)
    f3_ct = x
    #enc3
    x = Conv3D( 256 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(x)
    f4_ct = x
    #enc4
    x = Conv3D( 512 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(x)#valid->same
    f5_ct = x

    #model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))///////////////////////////////////////////////


    #PT
    input_img2 = Input( shape=( 128, 128, 48, 1) , name="inputs2_name") #  32×32、RGB
    #enc0
    y = Conv3D( 32 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(input_img2)
    f1_pt = y
    #enc1
    y = Conv3D( 64 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    y = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(y)
    f2_pt = y
    #enc2
    y = Conv3D( 128 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    y = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(y)
    f3_pt = y
    #enc3
    y = Conv3D( 256 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    y = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(y)
    f4_pt = y
    #enc4
    y = Conv3D( 512 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    y = MaxPooling3D(pool_size= 2, strides= 2 , padding='same')(y)#valid->same
    f5_pt = y


    #fusion/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fusion_pre = concatenate([f5_ct, f5_pt], axis=4)
    X = Conv3D( 512 , 1 , strides= 1 , activation = 'relu' ,padding='same' )(fusion_pre)
    X = BatchNormalization()(X)
    x_fusion_post = X
    #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    #CT
    #decode4
    #x = UpSampling2D(( 2, 2))(x_fusion_post)
    #f4_ctpt = concatenate([f4_ct, f4_pt], axis = 3)
    #tf.keras.layers.Conv2DTranspose(     filters,     kernel_size,     strides=(1, 1),     padding='valid',     output_padding=None,     data_format=None,     dilation_rate=(1, 1),     activation=None,     use_bias=True,     kernel_initializer='glorot_uniform',     bias_initializer='zeros',     kernel_regularizer=None,     bias_regularizer=None,     activity_regularizer=None,     kernel_constraint=None,     bias_constraint=None,     **kwargs )
    x = Conv3DTranspose( 512 , 2 , strides= 2 , padding= 'same') (x_fusion_post)
    x = concatenate([x, f4_ct, f4_pt], axis = 4)
    x = Conv3D( 256 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    #print(x.shape)
    #dec3
    x = Conv3DTranspose( 256 , 2 , strides= 2 , padding= 'same')(x)
    x = concatenate([x, f3_ct, f3_pt], axis = 4)
    x = Conv3D( 128 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    #print(x.shape)
    #dec2
    x = Conv3DTranspose( 128 , 2 , strides= 2 , padding= 'same')(x)
    x = concatenate([x, f2_ct, f2_pt], axis = 4)
    x = Conv3D( 64 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    #print(x.shape)
    #dec1
    x = Conv3DTranspose( 64 , 2 , strides= 2 , padding= 'same')(x)
    x = concatenate([x, f1_ct, f1_pt], axis = 4)
    x = Conv3D( 32 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(x)
    x = BatchNormalization()(x)
    #print(x.shape)

    #conv_cls
    #????
    logits_ct = Conv3D( 2 , 1 , strides= 1 , activation = 'relu' ,padding='same' )(x)#引数足りない////////////////////////////////////////////////////////////////////////////
    #print(logits_ct.shape)


    #PT
    #decode4
    y = Conv3DTranspose( 512 , 2 , strides= 2 , padding= 'same')(x_fusion_post)#////////////////////////////////////x
    y = concatenate([y, f4_pt, f4_ct], axis = 4)
    y = Conv3D( 256 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    #print(x.shape)
    #dec3
    y = Conv3DTranspose( 256 , 2 , strides= 2 , padding= 'same')(y)
    y = concatenate([y, f3_pt, f3_ct], axis = 4)
    y = Conv3D( 128 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    #print(x.shape)
    #dec2
    y = Conv3DTranspose( 128 , 2 , strides= 2 , padding= 'same')(y)
    y = concatenate([y, f2_pt, f2_ct], axis = 4)
    y = Conv3D( 64 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    #print(x.shape)
    #dec1
    y = Conv3DTranspose( 64 , 2 , strides= 2 , padding= 'same')(y)
    y = concatenate([y, f1_pt, f1_ct], axis = 4)
    y = Conv3D( 32 , 3 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    y = BatchNormalization()(y)
    #print(x.shape)

    #conv_cls
    #logitsは「ソフトマックス活性化関数に通す前のニューラルネットワークの出力」のこと
    logits_pt = Conv3D( 2 , 1 , strides= 1 , activation = 'relu' ,padding='same' )(y)
    #print(logits_pt.shape)
    
    
  #pred_ct
    #y_prob = Softmax(logits_ct, axis=-1)
    y_prob = Softmax(logits_ct)
  #pred_pt
    #y_prob2 = Softmax(logits_pt, axis=-1)
    y_prob2 = Softmax(logits_pt)
    
    #argmaxは配列の一番大きい値を返す
    #y_ct = np.argmax(logits_ct, axis=-1)
    #y_ct = tf.math.argmax(logits_ct, axis=-1)
    #y_ct = tf.keras.backend.argmax(logits_ct, axis=-1)
    y_ct = Conv3D( 1 , 1 , strides= 1 , activation = 'relu' ,padding='same' , name="ct_output")(x)
    #y_ct = Conv3D( 1 , 1 , strides= 1 , activation = 'relu' ,padding='same' , name="ct_output")(logits_ct)
    y_ct = tf.cast(y_ct, tf.float32)#, name="ct_output")
    
    #argmaxは配列の一番大きい値を返す
    #y_pt = tf.math.argmax(logits_pt, axis=-1)
    #y_pt = tf.keras.backend.argmax(logits_pt, axis=-1)
    y_pt = Conv3D( 1 , 1 , strides= 1 , activation = 'relu' ,padding='same'  , name="pt_output")(y)
    #y_pt = Conv3D( 1 , 1 , strides= 1 , activation = 'relu' ,padding='same' , name="pt_output")(logits_pt)
    y_pt = tf.cast(y_pt, tf.float32)#, name="pt_output")
    #print('y_ct',y_ct.shape)
    #print('y_pt',y_pt.shape)
    
    #print('input1',input_img.shape)
    #print('input2',input_img2.shape)


    """
        モデルのコンパイル
    """
    #model = Model( input_img, output_img )
    model = Model(inputs=[input_img, input_img2], outputs=[y_ct, y_pt], name="3DDFCN-Coseg_model")
    #print('modelinputshape',model.input_shape)
    #print('modeloutputshape',model.output_shape)
    #print('modelinput',model.input)
    #print('modelinput',model.output)
    
    #model.compile( optimizer='adam',loss=outer_loss(lossmap),metrics=["accuracy"] )
    
    #model.compile( optimizer='adam',loss='sparse_softmax_cross_entropy_with_logits',metrics=["accuracy"] )
    #model.compile( optimizer='adam',loss={ 'binary_crossentropy': 'ct_output', 'binary_crossentropy':'pt_output'} , loss_weights={'ct_output':'0.5', 'pt_output':'0.5'})#, metrics=['accuracy'])
    model.compile( optimizer='adam',loss=[ 'mse', 'mse' ])
    #dice_loss = sm.losses.DiceLoss()
    #metrics = [sm.metrics.IOUScore(threshold=0.5)]
    #model.compile( optimizer='adam',loss=[ dice_loss, dice_loss])#,  metrics=metrics)
    #model.compile( optimizer='adam',loss=[ tversky_loss, tversky_loss ])
    #autoencoder.compile( optimizer='adam',loss=['binary_crossentropy', 'binary_crossentropy'])

    #  アーキテクチャの可視化
    model.summary() #  ディスプレイ上に表示
    #plot_model( model, to_file="result3D/architecture.png" )
    #plot_model(model, show_shapes=True)


        #モデルの学習

    csv_logger = CSVLogger("result3D/training_3D"+name+".csv")
    #keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stop = EarlyStopping( monitor="val_loss", mode="auto" )
    tensor_board = TensorBoard( "./logs",
                    histogram_freq=0,
                    write_graph=True,
                        write_images=True ) 

    #keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    #check_point = ModelCheckpoint( filepath="./model.{epoch:02d}-{val_loss:.4f}.hdf5",
    check_point = ModelCheckpoint( filepath="result3D/model-dfcn_3D-{epoch:02d}"+name+".hdf5",
                       monitor="loss",
                       save_best_only=False,
                       period=1000,#何回ごとにモデルを保存するか
                       mode="auto" )

    cb = [ csv_logger, check_point ]


    epochs = 5000
    #epochs = 3

    #  学習
    #history = model.fit( [X_train, Xx_train], [Y_train, Yy_train], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=[X_train, Xx_train], [Y_train, Yy_train], callbacks=cb )
    #Xtrain=CT, Xxtrain=PET
    history = model.fit([X_train, Xx_train],[Y_train, Yy_train], epochs=epochs, batch_size=1, shuffle=True,  callbacks=cb) #batch_sizeデフォルトは32
    #history = model.fit([X_train, Xx_train],[Y_train, Yy_train], epochs=epochs, batch_size=None, callbacks=cb)

    model.save('result3D/modelsave_3D_dfcn'+name+'.h5')

    #  学習のグラフ化
    plot_history( history, epochs )

    #  画像の表示
    train_ct_img, train_pt_img = model.predict( [X_train ,Xx_train] )
    #train_pt_img = model.predict( Xx_train )

    #filePath =  './result'
    filePath = 'result3D'

    n = 4
    for i in range(n):
        #  現画像
        filename = filePath + '/test/traindataX' + str(i) + ''+name+'.raw'
        X_train[i].reshape( 128, 128, 48, 1).tofile(filename)

        filename = filePath + '/test/traindataXx' + str(i) + ''+name+'.raw'
        Xx_train[i].reshape( 128, 128, 48, 1).tofile(filename)
        #plt.gray()
        #  結果画像
        filename = filePath + '/kekka/resultCT3D_' + str(i) + ''+name+'.raw'#/////////////////////////////////////////////////////////////256*256
        train_ct_img[i].reshape( 128, 128, 48, 1).tofile(filename)

        filename = filePath + '/kekka/resultPET3D_' + str(i) + ''+name+'.raw'
        train_pt_img[i].reshape( 128, 128, 48, 1).tofile(filename)
        plt.gray()
        #教師データ
        filename = filePath + '/GT/teachdataY' + str(i) + ''+name+'.raw'
        Y_train[i].reshape( 128, 128, 48, 1).tofile(filename)

        filename = filePath + '/GT/teachdataYy' + str(i) + ''+name+'.raw'
        Yy_train[i].reshape( 128, 128, 48, 1).tofile(filename)
        #plt.gray()


if __name__ == '__main__':
    #tf.config.list_physical_devices("GPU").__len__() > 0
    #データの取得
    img_num = 4
    img_slice =48
    #投影サイズ
    pro_width = 128
    pro_height = 128
    pro_axis = 1
    #教師データサイズs
    teach_width = 128
    teach_height = 128
    teach_axis = 1
    #入力画像読み込み
    #CT,PETはdicom_process,,,,,,,,GroundTruthは作成したrawデータからの入力8bitからfroat32への変換必要

    #PET,CTもrawから

    #ct
    input_dir = '3DDFCNdatasets_3/CT_input128/*.raw'    
    file_list = sorted(glob.glob(input_dir, recursive = True), key=natural_keys)
    #file_list = glob.glob(input_dir, recursive=True)

    for i in range(len(file_list)):
        width=128
        height=128
        fd = open(file_list[i], 'rb')
        f = np.fromfile(fd, dtype=np.float32, count=height*width)
        img_input = f.reshape((height,width))
        fd.close()
        img_input = np.clip( img_input, 0., 1. ) #0～1の範囲に納める
        image_list.append(img_input)

    #pet
    input_dir = '3DDFCNdatasets_3/PET_input128/*.raw'    
    file_list = sorted(glob.glob(input_dir, recursive = True), key=natural_keys)
    #file_list = glob.glob(input_dir, recursive=True)

    for i in range(len(file_list)):
        width=128
        height=128
        fd = open(file_list[i], 'rb')
        f = np.fromfile(fd, dtype=np.float32, count=height*width)
        img_input2 = f.reshape((height,width))
        fd.close()
        img_input2 = np.clip( img_input2, 0., 1. )
        image_list2.append(img_input2)



    #教師データ(ラベリング画像)読み込み
    #ct//////////////MRI
    input_dir = '3DDFCNdatasets_3/MRI_label_black/*.raw'    
    file_list = sorted(glob.glob(input_dir, recursive = True), key=natural_keys)
    #file_list = glob.glob(input_dir, recursive=True)

    for i in range(len(file_list)):
        width=128
        height=128
        fd = open(file_list[i], 'rb')
        f = np.fromfile(fd, dtype=np.uint8, count=height*width)
        f = f.astype('float32')
        img_label = f.reshape((height,width))
        fd.close()
        #img_label = zscore(img_label)
        #img_label = img_label.astype('float32')
        img_label = np.clip( img_label, 0., 1. )
        label_list.append(img_label)


    #pt
    input_dir = '3DDFCNdatasets_3/PET_label_black/*.raw'    
    file_list = sorted(glob.glob(input_dir, recursive = True), key=natural_keys)
    #file_list = glob.glob(input_dir, recursive=True)

    for i in range(len(file_list)):
        width=128
        height=128
        fd = open(file_list[i], 'rb')
        f = np.fromfile(fd, dtype=np.uint8, count=height*width)
        f = f.astype('float32')
        img_label2 = f.reshape((height,width))
        fd.close()
        #img_label2 = zscore(img_label2)
        #img_label2 = img_label2.astype('float32')
        img_label2 = np.clip( img_label2, 0., 1. )
        label_list2.append(img_label2)


    # kerasに渡すためにnumpy配列に変換。
    image_list = np.array(image_list)
    img_3D = np.split(image_list, img_num)
    img_3D = np.array(img_3D)
    image_list2 = np.array(image_list2)
    img_3D_2 = np.split(image_list2, img_num)
    img_3D_2 = np.array(img_3D_2)
    label_list = np.array(label_list)
    label_3D = np.split(label_list, img_num)
    label_3D = np.array(label_3D)
    label_list2 = np.array(label_list2)
    label_3D_2 = np.split(label_list2, img_num)
    label_3D_2 = np.array(label_3D_2)

    # 学習用データとテストデータ
    #X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0, random_state=1)
        
        
    X_train=img_3D.reshape( int( img_num * 1.0 ) , pro_width, pro_height, int(img_slice), 1 ) #画像の枚数＊横＊縦*深さ＊チャンネル数
    Y_train=label_3D.reshape( int( img_num * 1.0 ) , teach_width, teach_height, int(img_slice), 1) 
    Xx_train=img_3D_2.reshape( int( img_num * 1.0 ) , pro_width, pro_height, int(img_slice), 1 ) 
    Yy_train=label_3D_2.reshape( int( img_num * 1.0 ) , teach_width, teach_height, int(img_slice), 1 ) 
    #print('X',X_train.shape)
    #print('Xx',Xx_train.shape)
    #print('Y',Y_train.shape)
    #print('Yy',Yy_train.shape)
    

    #  モデル
    DFCN( X_train, Xx_train, Y_train, Yy_train)
