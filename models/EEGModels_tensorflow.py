
# import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Activation,  Dropout  # Permute,
from keras.layers import Conv2D,  AveragePooling2D, GlobalAveragePooling2D  # MaxPooling2D,
# from keras.layers import Conv1D, Lambda, Add
import keras.initializers.initializers_v2
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Concatenate  # Flatten,
from keras.constraints import max_norm
# from keras import backend as K

# ========================================================================================
# MI_EEGNet
# ========================================================================================
def MI_EEGNet(nb_classes,Chans=22,Samples=64,dropoutRate=0.5):     # dropoutRate=0.2-0.5

    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(64,(1,16),padding='same',
                    input_shape=(Chans, Samples, 1), kernel_initializer='glorot_uniform')(input_main)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), padding='valid',
                             depth_multiplier=4,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)  # Dropout

    #inceptions结构
    block2 = Conv2D(64,(1,1),padding='same',  kernel_initializer='glorot_uniform')(block1)
    block2 = SeparableConv2D(64,(1,7),padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(dropoutRate)(block2)
    block2 = SeparableConv2D(64, (1, 7), padding='same')(block2)
    block2 = AveragePooling2D((1, 2))(block2)

    block3 = Conv2D(64, (1, 1), padding='same', kernel_initializer='glorot_uniform')(block1)
    block3 = SeparableConv2D(64, (1, 9), padding='same')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(dropoutRate)(block3)
    block3 = SeparableConv2D(64, (1, 9), padding='same')(block3)
    block3 = AveragePooling2D((1, 2))(block3)

    block4 = AveragePooling2D((1, 2))(block1)
    block4 = Conv2D(64,(1,1),padding='same', kernel_initializer='glorot_uniform')(block4)

    block5 = Conv2D(64,(1,1),padding='same',
                    strides=(1,2), kernel_initializer='glorot_uniform')(block1)

    block = Concatenate(axis=-1)([block2, block3, block4, block5])

    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = SeparableConv2D(256,(1,5),padding='same')(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(dropoutRate)(block)

    block = GlobalAveragePooling2D()(block)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(block)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# def MI_EEGNet(nb_classes=4,Chans=22,Samples=256,dropoutRate=0.3):
#
#     input_main = Input((Chans, Samples, 1))
#     block1 = Conv2D(64,(1,16),padding='same',
#                     input_shape=(Chans, Samples, 1))(input_main)
#     block1 = BatchNormalization()(block1)
#     block1 = DepthwiseConv2D((Chans, 1),padding='valid',
#                              depth_multiplier=4,
#                              depthwise_constraint=max_norm(1.))(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Activation('elu')(block1)
#     block1 = AveragePooling2D((1, 2))(block1)
#     block1 = Dropout(dropoutRate)(block1)
#
#     #inceptions结构
#     block2 = Conv2D(64,(1,1),padding='same')(block1)
#     block2 = SeparableConv2D(64,(1,7),padding='same')(block2)
#     block2 = BatchNormalization()(block2)
#     block2 = Activation('elu')(block2)
#     block2 = Dropout(dropoutRate)(block2)
#     block2 = SeparableConv2D(64, (1, 7), padding='same')(block2)
#     block2 = AveragePooling2D((1, 2))(block2)
#
#     block3 = Conv2D(64, (1, 1), padding='same')(block1)
#     block3 = SeparableConv2D(64, (1, 9), padding='same')(block3)
#     block3 = BatchNormalization()(block3)
#     block3 = Activation('elu')(block3)
#     block3 = Dropout(dropoutRate)(block3)
#     block3 = SeparableConv2D(64, (1, 9), padding='same')(block3)
#     block3 = AveragePooling2D((1, 2))(block3)
#
#     block4 = AveragePooling2D((1, 2))(block1)
#     block4 = Conv2D(64,(1,1),padding='same')(block4)
#
#     block5 = Conv2D(64,(1,1),padding='same',
#                     strides=(1,2))(block1)
#
#     block = Concatenate(axis=-1)([block2, block3, block4, block5])
#
#     block = BatchNormalization()(block)
#     block = Activation('elu')(block)
#     block = SeparableConv2D(256,(1,5),padding='same')(block)
#     block = BatchNormalization()(block)
#     block = Activation('elu')(block)
#     block = Dropout(dropoutRate)(block)
#
#     block = GlobalAveragePooling2D()(block)
#
#     dense = Dense(nb_classes,kernel_constraint=max_norm(0.5))(block)
#     softmax = Activation('softmax')(dense)
#
#     return Model(inputs=input_main, outputs=softmax)


if __name__ == '__main__':
    #  1: ShallowConvNet   2:DeepConvNet   3:EEGNet8-2
    #  4:EEGNet4-2         5:EEG_TCNet     6:EEG_inception
    #  7:EEG_ITNet         8:MI_EEGNet
    num = 8
    nb_classes = 4

    # if num == 1:
    #     model = ShallowConvNet(nb_classes)
    # elif num == 2:
    #     model = DeepConvNet(nb_classes)
    # elif num == 3:
    #     model = EEGNet(nb_classes,dropoutRate=0.5,kernLength=32,F1=8,D=2, F2=16)
    # elif num == 4:
    #     model = EEGNet(nb_classes,dropoutRate=0.5,kernLength=32,F1=4, D=2, F2=8)
    # elif num == 5:
    #     model = EEG_TCNet(nb_classes)
    # elif num == 6:
    #     model = EEG_inception(nb_classes)
    # elif num == 7:
    #     model = EEG_ITNet(nb_classes)
    if num == 8:
        model = MI_EEGNet(nb_classes, Chans=22, Samples=256)
    # elif num == 9:
    #     model = mymodel(nb_classes)

    model.summary()
    numParams = model.count_params()
    print(numParams)

