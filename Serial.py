from scipy import signal
from scipy import io
from scipy import random
import os
import serial
import sys
import numpy as np
import time
import struct
from dnnCS_functions import *

# The data, split between train and test sets:
# Load y_test float32 type
data = io.loadmat('float_y.mat')
y_test = data['ans'][:32,:].T
# sparsity basis
D = io.loadmat('D.mat')['D']
# sensing matrix
A = io.loadmat('A.mat')['A'][:32,:]
# dataset
data = io.loadmat('test_set.mat')
X_test, S_test = data['X_test'], data['S_test']
B = A@D

#Check the correct COMx port associated to the STLink and choose a consistend baud rate
port = 'COM6'
baud = 115200  
ser = serial.Serial(port, baud, timeout=0)

for i in range(100):
    for j in range(32):
        #Iterations over the 32 elements of an y_test vector
        ser.write(bytearray(y_test[i][j]))
        time.sleep(.1)
    ser.flushOutput()
    #Reading the sent data. Note the output is an hexadecimal string of 128 bytes filled with 0s and 1s
    reading = ser.readline()
    ser.flushOutput()
    #Converting the data into an numpy boolean array type, required by xi_estimation function
    s_hat = struct.unpack('????????????????????????????????????????????????????????????????', reading)
    s_hat = np.asarray(s_hat)
    xi = xi_estimation(y_test[i], s_hat, B)
    x_hat = D@xi
    print('RSNR = {r:6.2f} with support missmatch = {ms:d}'.format(r=RSNR(X_test[i],x_hat),ms=int(sum(np.abs(S_test[i]-s_hat)))))
    
ser.close()
