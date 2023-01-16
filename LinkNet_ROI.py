
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import keras
import glob
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
layers = keras.layers

USE_BTCHNORM = True

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def my_dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.''' 
    FP = False_Pos(y_true, y_pred) 
    FN=False_Neg(y_true, y_pred)
    TP=True_Pos(y_true, y_pred)
    Dice= ((2.0 * TP + smooth) / ((2.0 * TP)+FP+FN + smooth))
    return Dice

def False_Neg(y_true, y_pred):
    ''' False Negative (FN): we predict a label of 0 (negative), but the true label is 1..'''
    batch_size=10
    a = np.ones((batch_size,512,512,1))
    y_pred1 = a - y_pred
    FN = K.sum(y_true*y_pred1,axis=(1,2,3))
    return FN/batch_size
    
def False_Pos(y_true, y_pred): 
    ''' False Positive (FP): we predict a label of 1 (positive), but the true label is 0.'''
    batch_size=10
    a = np.ones((batch_size,512,512,1))
    y_true1 = a - y_true
    FP = K.sum(y_true1*y_pred,axis=(1,2,3))     
    return FP/batch_size
def True_Pos(y_true, y_pred):
    ''' True Positive (TP): we predict a label of 1 (positive), and the true label is 1.'''
    batch_size=10
    TP = K.sum(y_true*y_pred,axis=(1,2,3))
    return TP/batch_size
def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=1.0)
    

def loss_fcn(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    #dice_term = K.log(dice_coef(y_true, y_pred, 1.0))
    dice_term = K.exp(1 + dice_coef(y_true, y_pred, 1.0))
    return 1 - dice_term


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    return mvn


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = layers.Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=keras.regularizers.l2(0.0001))(input)

    return layers.add([shortcut, residual])


def encoder_block(input_tensor, m, n):

    kwargs = dict(
        padding='same',
        use_bias=True,
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        kernel_regularizer= None, #l2(l=weight_decay),
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )    
    
    x = layers.BatchNormalization()(input_tensor) if USE_BTCHNORM else layers.Lambda(mvn)(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), **kwargs)(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=n, kernel_size=(3, 3), **kwargs)(x)

    added_1 = _shortcut(input_tensor, x)

    x = layers.BatchNormalization()(added_1) if USE_BTCHNORM else layers.Lambda(mvn)(added_1)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=n, kernel_size=(3, 3), **kwargs)(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=n, kernel_size=(3, 3), **kwargs)(x)

    added_2 = _shortcut(added_1, x)

    return added_2


def decoder_block(input_tensor, m, n):

    kwargs = dict(
        padding='same',
        use_bias=True,
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        kernel_regularizer= None, #l2(l=weight_decay),
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )        
    
    x = layers.BatchNormalization()(input_tensor) if USE_BTCHNORM else layers.Lambda(mvn)(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=int(m/4), kernel_size=(1, 1), **kwargs)(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=int(m/4), kernel_size=(3, 3), **kwargs)(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=n, kernel_size=(1, 1), **kwargs)(x)

    return x


def LinkNet(input_shape=(512, 512, 1), classes=1, weights=None):
    inputs = layers.Input(shape=input_shape)
    inputs_normalized = layers.Lambda(mvn)(inputs) 
    x = layers.BatchNormalization()(inputs_normalized) if USE_BTCHNORM else layers.Lambda(mvn)(inputs_normalized)
    x = layers.Activation('relu')(inputs_normalized)
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    encoder_1 = encoder_block(input_tensor=x, m=64, n=64)
    encoder_2 = encoder_block(input_tensor=encoder_1, m=64, n=128)
    encoder_2 = layers.SpatialDropout2D(0.5)(encoder_2) # Dropout
    encoder_3 = encoder_block(input_tensor=encoder_2, m=128, n=256)
    encoder_4 = encoder_block(input_tensor=encoder_3, m=256, n=512)
    encoder_4 = layers.SpatialDropout2D(0.5)(encoder_4) # Dropout
    decoder_4 = decoder_block(input_tensor=encoder_4, m=512, n=256)
    decoder_3_in = layers.add([decoder_4, encoder_3])
    decoder_3_in = layers.Activation('relu')(decoder_3_in)
    decoder_3 = decoder_block(input_tensor=decoder_3_in, m=256, n=128)
    decoder_3 = layers.SpatialDropout2D(0.5)(decoder_3) # Dropout
    decoder_2_in = layers.add([decoder_3, encoder_2])
    decoder_2_in = layers.Activation('relu')(decoder_2_in)
    decoder_2 = decoder_block(input_tensor=decoder_2_in, m=128, n=64)
    decoder_1_in = layers.add([decoder_2, encoder_1])
    decoder_1_in = layers.Activation('relu')(decoder_1_in)
    decoder_1 = decoder_block(input_tensor=decoder_1_in, m=64, n=64)
    decoder_1 = layers.SpatialDropout2D(0.5)(decoder_1) # Dropout
    # mask output
    x = layers.UpSampling2D((2, 2))(decoder_1)
    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=classes, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='glorot_normal',
                      name='mask_output')(x)
    final_output = x
    model = keras.models.Model(inputs=inputs, outputs=final_output)
    if weights is not None:
        model.load_weights(weights)
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=loss_fcn, metrics=[dice_coef, mean_iou])
    return model
    
my_model = LinkNet()
my_model.summary()

# Train the model
filepath_best = 'C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Logs/ROI' + "val_loss_{epoch:2d}-{val_loss:.2f}.h5"
best_check = ModelCheckpoint(filepath_best, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=10)

filepath_best2 = 'C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Logs/ROI' + "val_dice_coef_{epoch:2d}-{val_dice_coef:.2f}.h5"
best_check2 = ModelCheckpoint(filepath_best2, monitor='val_dice_coef', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

history= my_model.fit (FinaltrainData, FinaltrainMask ,validation_split=0.2 , batch_size=10 , epochs=100, shuffle = 'true',verbose=1,callbacks=[best_check,best_check2])
score=my_model.evaluate(FinaltrainData, FinaltestMask, batch_size=10)

print("Test:accuaracy", str(score[1]*100))
print("Test:Total loss",str(score[0]*100))
print(score)
print(history.history.keys())   
# summarize history for dice
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for iou
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.title('mean_iou')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()