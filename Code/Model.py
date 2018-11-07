from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D, Add, Input, Dense, Concatenate
from keras.optimizers import SGD, RMSprop, Adadelta,Adam
from keras.models import Model
from keras import backend as K
from Loss import mean_squared_error_weight
def conv3(x,nb_filter,pad = 'same'):
    y = Convolution3D(nb_filter, (3, 3, 3), padding=pad, use_bias=False)(x)
    return Activation('relu')(y)


def dense_block(x, nb_filter):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x1 = conv3(x,nb_filter)
    x11 = Concatenate(axis=concat_axis)([x,x1])
    x2 = conv3(x11,nb_filter)
    x22 = Concatenate(axis=concat_axis)([x,x1,x2])
    x3 = conv3(x22,nb_filter,'valid')
    return x3

def single_instance(x0):
    x1 = dense_block(x0,16)
    x1 = MaxPooling3D(pool_size=(2, 2, 2))(x1)

    x2 = dense_block(x1,32)
    x2 = MaxPooling3D(pool_size=(2, 2, 2))(x2)

    x3 = dense_block(x2,64)
    x3 = MaxPooling3D(pool_size=(2, 2, 2))(x3)

    fc0 = Flatten()(x3)
    fc1 = Dense(32)(fc0)
    fc2 = Dense(8)(fc1)
    return fc2


def merged_model(patchsize, landmk_num, numofscales):
    branch = []
    input_all = []
    for i_num in range(0, landmk_num):
        input_all.append(Input(shape = (numofscales, patchsize, patchsize, patchsize),name = 'input_{0}'.format(i_num)))
        branch.append(single_instance(input_all[i_num]))
    merged = Concatenate(axis=-1)(branch)
    m_fc1 = Dense(128,activation='relu',name='FC1')(merged)
    m_fc2 = Dense(32*landmk_num,activation='relu',name='FC2')(m_fc1)
    m_output = Dense(16,name = 'output_final')(m_fc2)

    merge_model = Model(inputs = input_all, outputs = m_output )
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    merge_model.compile(loss=mean_squared_error_weight, optimizer=sgd)

    return merge_model
