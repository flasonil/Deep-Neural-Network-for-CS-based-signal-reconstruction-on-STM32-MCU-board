import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer, Dense
#from dnnCS_layers import Antipodal

#############################
### custom network ##########
#############################

def dnnDec(input_0, n):
    ### Level 0
    out_0 = Dense(units=2*n, activation='relu', name='H0')(input_0)
    ### Level 1
    out_1 = Dense(units=2*n, activation='relu', name='H1')(out_0)
    ### Level 2
    out_2 = Dense(units=n, activation='relu', name='H2')(out_1)
    ### Final output
    out_nn = Dense(units=n, activation='sigmoid', name='Out')(out_2)
    return out_nn


def dnnAntipodal(input_0, n, m):
    ### CS encoder
    y_0 = Antipodal(units=m, input_shape=(n,), name='A')(input_0)
    ### Level 0
    out_0 = Dense(units=2*n, activation='relu', name='H0')(y_0)
    ### Level 1
    out_1 = Dense(units=2*n, activation='relu', name='H1')(out_0)
    ### Level 2
    out_2 = Dense(units=n, activation='relu', name='H2')(out_1)
    ### Final output
    out_nn = Dense(units=n, activation='sigmoid', name='Out')(out_2)
    return out_nn


def dnnFloat(input_0, n, m):
    ### CS encoder
    y_0 = Dense(units=m, activation='linear', use_bias=False, name='A')(input_0)
    ### Level 0
    out_0 = Dense(units=2*n, activation='relu', name='H0')(y_0)
    ### Level 1
    out_1 = Dense(units=2*n, activation='relu', name='H1')(out_0)
    ### Level 2
    out_2 = Dense(units=n, activation='relu', name='H2')(out_1)
    ### Final output
    out_nn = Dense(units=n, activation='sigmoid', name='Out')(out_2)
    return out_nn

def dnnFloatFake(input_0, n, m):
    ### CS encoder
    y_0 = Dense(units=m, activation='linear', name='MatA')(input_0)
    ### Level 0
    out_0 = Dense(units=2*n, activation='relu', name='H0')(y_0)
    ### Level 1
    out_1 = Dense(units=2*n, activation='relu', name='H1')(out_0)
    ### Level 2
    out_2 = Dense(units=n, activation='relu', name='H2')(out_1)
    ### Final output
    out_nn = Dense(units=n, activation='sigmoid', name='Out')(out_2)
    return out_nn

