import scipy.io as sio
import h5py
from keras_preprocessing import image
from __future__ import absolute_import
from __future__ import print_function
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.layers import Concatenate , concatenate
import numpy as np
import tensorflow as tf
import keras
from keras.utils import plot_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import scipy as sp
layers = keras.layers
USE_BTCHNORM = True
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
   # intersection = K.sum(y_true * y_pred, axis=axes)
   # summation = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    B= (2.0 * TP)+FP+FN + smooth
    A= (2.0 * TP + smooth) 
    Dice= A/B
    return Dice

def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def TP(y_true, y_pred):
    ''' True Positive (TP): we predict a label of 1 (positive), and the true label is 1.'''
    #TP=np.sum(np.logical_and(y_pred == 1, y_true == 1)) 
    TP=np.sum(keras.backend.all(keras.backend.stack([y_pred == 1, y_true == 1], axis=0)))
    return TP

def TN(y_true, y_pred):
    ''' True Negative (TN): we predict a label of 0 (negative), and the true label is 0.'''
    TN=np.sum(keras.backend.all(keras.backend.stack([y_pred == 0, y_true == 0], axis=0)))
    #TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    return TN

def FP(y_true, y_pred):
    ''' False Positive (FP): we predict a label of 1 (positive), but the true label is 0.'''
   # FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FP=np.sum(keras.backend.all(keras.backend.stack([y_pred == 1, y_true == 0], axis=0)))
    return FP

def FN(y_true, y_pred):
    ''' False Negative (FN): we predict a label of 0 (negative), but the true label is 1..'''
    FN=np.sum(keras.backend.all(keras.backend.stack([y_pred == 0, y_true == 1], axis=0)))
   # FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return FN

def False_Neg(y_true, y_pred):
    batch_size=10
    a = np.ones((batch_size,512,512,1))
#        y_true = y_true//255
    y_pred1 = a - y_pred
    FN = K.sum(y_true*y_pred1,axis=(1,2,3))
#        FN = FN/(input_size[0]*input_size[1])
    return FN/batch_size
    
def False_Pos(y_true, y_pred): 
    batch_size=10
    a = np.ones((batch_size,512,512,1))
#        y_true = y_true//255
    y_true1 = a - y_true
    FP = K.sum(y_true1*y_pred,axis=(1,2,3))
#        FP = FP/(input_size[0]*input_size[1])      
    return FP/batch_size
def True_Pos(y_true, y_pred):
    batch_size=10
#        y_true = y_true//255
    TP = K.sum(y_true*y_pred,axis=(1,2,3))
    return TP/batch_size
def Summation (y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)   
    return summation

def intersection (y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)  
    return intersection

def dice_coef_loss(y_true, y_pred):
    return 1.0 - my_dice_coef(y_true, y_pred, smooth=1)
    
def loss_fcn(y_true, y_pred):  
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
      #dice_term = K.log(dice_coef(y_true, y_pred, 1.0))
    dice_term = K.exp(1 + my_dice_coef(y_true, y_pred,0))
    return (bce - dice_term)

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
        kernel_regularizer=None,  # l2(l=weight_decay),
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
        kernel_regularizer=None,  # l2(l=weight_decay),
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )

    x = layers.BatchNormalization()(input_tensor) if USE_BTCHNORM else layers.Lambda(mvn)(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=int(m / 4), kernel_size=(1, 1), **kwargs)(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=int(m / 4), kernel_size=(3, 3), **kwargs)(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=n, kernel_size=(1, 1), **kwargs)(x)

    return x


def Multi_Task_Brain(input_shape=(256, 256, 2),input_shape2=(256,256,1), classes=2, num_classes=3, weights=None,batch_size=10):
    inputs = layers.Input(shape=input_shape)
    inputs_normalized = layers.Lambda(mvn)(inputs)  # whitening of input data
    x_half = layers.MaxPooling2D((2,2),strides=(2,2))(inputs_normalized)
    x_half_half = layers.MaxPooling2D((2,2),strides=(2,2))(x_half)
    x_half_half_half = layers.MaxPooling2D((2,2),strides=(2,2))(x_half_half)
    
    inputs1 = layers.Input(shape=input_shape2)
    inputs_normalized1 = layers.Lambda(mvn)(inputs1)  # whitening of input data
    m_half = layers.MaxPooling2D((2,2),strides=(2,2))(inputs_normalized1)
    m_half_half = layers.MaxPooling2D((2,2),strides=(2,2))(m_half)
    m_half_half_half = layers.MaxPooling2D((2,2),strides=(2,2))(m_half_half)
    
    x = layers.BatchNormalization()(inputs_normalized) if USE_BTCHNORM else layers.Lambda(mvn)(inputs_normalized)
    x = layers.Activation('relu')(inputs_normalized)
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = layers.Activation('relu')(inputs_normalized)
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x_half],axis=-1)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x_half_half],axis=-1)
    encoder_1 = encoder_block(input_tensor=x, m=64, n=64)
    encoder_1 = layers.concatenate([encoder_1, x_half_half_half])
    encoder_2 = encoder_block(input_tensor=encoder_1, m=64, n=128)
    encoder_2 = layers.SpatialDropout2D(0.5)(encoder_2)  # Dropout
    encoder_3 = encoder_block(input_tensor=encoder_2, m=128, n=256)
    encoder_4 = encoder_block(input_tensor=encoder_3, m=256, n=512)
    encoder_4 = layers.SpatialDropout2D(0.5)(encoder_4)  # Dropout


#start decoder---------------------
    decoder_4 = decoder_block(input_tensor=encoder_4, m=512, n=256)
    decoder_4 = layers.concatenate([decoder_4, m_half_half_half],axis=-1)
    decoder_4_2 = decoder_block(input_tensor=encoder_4, m=512, n=256)#cp

    decoder_3_in = layers.add([decoder_4_2, encoder_3])
    decoder_3_in = layers.Activation('relu')(decoder_3_in)
    decoder_3 = decoder_block(input_tensor=decoder_3_in, m=256, n=128)
    decoder_3 = layers.concatenate([decoder_3, m_half_half],axis=-1)
    decoder_3 = layers.SpatialDropout2D(0.5)(decoder_3)  # Dropout

    decoder_2_in = layers.add([decoder_3, encoder_2])
    decoder_2_in = layers.Activation('relu')(decoder_2_in)
    decoder_2 = decoder_block(input_tensor=decoder_2_in, m=128, n=64)
    decoder_2 = layers.concatenate([decoder_2, m_half],axis=-1)

    decoder_1_in = layers.add([decoder_2, encoder_1])
    decoder_1_in = layers.Activation('relu')(decoder_2)
    decoder_1 = decoder_block(input_tensor=decoder_1_in, m=64, n=64)
    decoder_1 = layers.SpatialDropout2D(0.5)(decoder_1)  # Dropout


    # mask output
    x = layers.UpSampling2D((2, 2))(decoder_1)
    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('softplus')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('softplus')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
    x = layers.UpSampling2D((2, 2), name='beforeconcat')(x)

    x = layers.BatchNormalization()(x) if USE_BTCHNORM else layers.Lambda(mvn)(x)
    x = layers.Activation('softplus')(x)
    x = layers.Conv2D(filters=classes, kernel_size=(3, 3), padding="same", activation='sigmoid',
                      kernel_initializer='glorot_normal',name='mask_output')(x)

    final_output = x
    
    # Classification
    encoder_1=layers.MaxPooling2D((2,2),strides=(2,2))(encoder_1)
    encoder_1=layers.MaxPooling2D((2,2),strides=(2,2))(encoder_1)
    Featuers_1=layers.MaxPooling2D((2,2),strides=(2,2))(encoder_1)
    encoder_2=layers.MaxPooling2D((2,2),strides=(2,2))(encoder_2)
    Featuers_2=layers.MaxPooling2D((2,2),strides=(2,2))(encoder_2)
    Aggregate =layers.concatenate([Featuers_1, Featuers_2],axis=-1)
    Featuers_3=layers.MaxPooling2D((2,2),strides=(2,2))(encoder_3)
    Aggregate =layers.concatenate([Aggregate, Featuers_3],axis=-1)
    Aggregate = layers.concatenate([Aggregate, encoder_4 ],axis=-1)
    flatten = layers.Flatten()(Aggregate)
    flatten = layers.Activation('relu')(flatten)
    fc1 = layers.Dense(1024,name='fc1')(flatten)
    fc1 = layers.Dense(1024,name='fc1')(fc1)
    fc0 = layers.BatchNormalization(name='fc1_bN')(fc1)
    fc0 = layers.Activation('relu',)(fc0)
    fc0 = layers.Dropout(0.3)(fc0)
    x2 = layers.Dense(num_classes,activation='sigmoid', name='classification')(fc0)
    final_output2 = x2
    


    model = keras.models.Model(inputs=[inputs,inputs], outputs=[final_output,final_output2])

    if weights is not None:
        model.load_weights(weights)

    ### Multi Task (Segmentation and classification)
    Metrics = {'segmentation': [dice_coef, mean_iou], 'classification': 'accuracy'}
    losses = {'segmentation': loss_fcn,'classification': 'categorical_crossentropy'}
    Loss_weights = {'segmentation': 2.0, 'classification':1.0}
    optimizer = keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer=optimizer, loss=losses, metrics=Metrics, loss_weights=Loss_weights,
                  sample_weight_mode='temporal')

    return model


my_model = Multi_Task_Brain()
my_model.summary()


# ----------------------------------
filepath_best1 = 'C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Logs/' + "val_segmentation_my_dice_coef{epoch:2d}-{val_segmentation_my_dice_coef:.2f}.h5"
best_check1 = ModelCheckpoint(filepath_best1, monitor='val_segmentation_my_dice_coef', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

filepath_best2 = 'C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Logs/' + "val_loss{epoch:2d}-{val_loss:.2f}.h5"
best_check2 = ModelCheckpoint(filepath_best2, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

filepath_best3 = 'C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Logs/' + "val_classification_categorical_accuracy{epoch:2d}-{val_classification_categorical_accuracy:.2f}.h5"
best_check3 = ModelCheckpoint(filepath_best2, monitor='val_classification_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')


Outputss={ 'segmentation': FinaltrainMask, 'classification':y}
history = model.fit([FinaltrainData, FinalMap] , Outputss ,validation_split=0.2 , batch_size=10, epochs=200, 
                     shuffle = 'true',verbose=1,callbacks=[best_check2,best_check1,best_check3])


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('dice_coef')
plt.legend()
plt.show()

history_dict = history.history
print(history_dict.keys())

loss = history.history['segmentation_my_dice_coef']
val_loss =   history.history['val_segmentation_my_dice_coef']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training my_dice_coef')
plt.plot(epochs, val_loss, color='green', label='Validation my_dice coef')
plt.title('Training and validation dice coef')
plt.xlabel('Epochs')
plt.ylabel('my_dice_coef')
plt.legend()
plt.show()

loss = history.history['segmentation_dice_coef']
val_loss =   history.history['val_segmentation_dice_coef']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training dice_coef')
plt.plot(epochs, val_loss, color='green', label='Validation dice coef')
plt.title('Training and validation dice coef')
plt.xlabel('Epochs')
plt.ylabel('my_dice_coef')
plt.legend()
plt.show()
























