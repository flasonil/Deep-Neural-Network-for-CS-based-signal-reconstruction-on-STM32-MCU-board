import numpy as np
import pandas as pd
import sys
from dnnCS_functions import *

from tensorflow.keras.models import Model


def find_threshold_AE(model, D, X, A=None, FoM='arsnr', isnr=np.inf, find_N=20, find_Iter=3, th_eps=1e-10, verbose=False):
    '''
    find threshold to be applied ot the DNN where the DNN model includes the CS encoder (auto encoder, AE)
    INPUT
    model: DNN model that includes the ENC-layer
    D: sparsity basis
    X: input vectors, ndarray
    A: matrix representing the ENC-layer
    FoM: figure of merit to be optimized
    isnr: level of additive white gaussian noise
    find_N: number of values for each discretization of the range ]0,1[ 
    find_Iter: number of iteration
    the_eps: eps and 1-eps are the initial external points
    verbose: flag for printing partial results
    
    OUTPUT
    th: threshold to be applied at the DNN output
    val: obtained optimum value
    '''
    if A is None:
        A = model.layers[1].get_weights()[0].T
        l_type = getattr(model.get_layer(index=0),'type_l','Base')
        if l_type=='Antipodal':
            A = np.sign(A)
    
    RSNRmin = min([150, isnr-5])
    
    Xn = add_awgn(X, isnr)
    B = np.dot(A, D)
    Y = np.dot(A, Xn.T).T
    O = model.predict(Xn)
    N = X.shape[0]
    
    rsnr = np.zeros(N,)
    fom_v = np.zeros(find_N,)
    
    for iL in range(find_Iter):
        if iL == 0:
            # log-scale search
            thV = np.logspace(-9, 0, find_N)
            thV[0] = thV[0] + th_eps
            thV[-1] = thV[-1] - th_eps
        else:
            # linear-scale search
            lb = thV[max([0, imax-1])]
            ub = thV[min([find_N-1, imax+1])]
            thV = np.linspace(lb, ub, find_N)
            
        for iTH in range(find_N):
            Sth = O > thV[iTH]
            for i in range(N):
                x, y, s = X[i,:], Y[i,:], Sth[i,:]
                xi = xi_estimation(y, s, B)
                rsnr[i] = RSNR(x, np.dot(D, xi))
            if FoM=='arsnr':
                fom_v[iTH] = np.mean(rsnr)
            elif FoM=='pcr':
                fom_v[iTH] = np.sum(rsnr>RSNRmin)/N
            else:
                raise ValueError('arsnr and pcr are acceptable FoMs')
        imax = np.argmax(fom_v)
        th = thV[imax]
        maxfom = fom_v[imax]
        if verbose:
            print(' => {:d}) ibest = {:d}, th = {:.6e}  ({} = {:.2f})'.format(iL, imax, th, FoM ,maxfom))
    return th, maxfom

def find_threshold(model, D, X, A, FoM='arsnr', isnr=np.inf, find_N=20, find_Iter=3, th_eps=1e-10, verbose=False):
    '''
    find threshold to be applied ot the DNN where the DNN model does NOT include the CS encoder
    INPUT
    model: DNN model that includes the ENC-layer
    D: sparsity basis
    X: input vectors, ndarray
    A: matrix representing the ENC-layer
    FoM: figure of merit to be optimized
    isnr: level of additive white gaussian noise
    find_N: number of values for each discretization of the range ]0,1[ 
    find_Iter: number of iteration
    the_eps: eps and 1-eps are the initial external points
    verbose: flag for printing partial results
    
    OUTPUT
    th: threshold to be applied at the DNN output
    val: obtained optimum value
    '''
    
    RSNRmin = min([150, isnr-5])
    
    Xn = add_awgn(X, isnr)
    B = np.dot(A, D)
    Y = np.dot(A, Xn.T).T
    O = model.predict(Y)
    N = X.shape[0]
    
    rsnr = np.zeros(N,)
    fom_v = np.zeros(find_N,)
    
    for iL in range(find_Iter):
        if iL == 0:
            # log-scale search
            thV = np.logspace(-9, 0, find_N)
            thV[0] = thV[0] + th_eps
            thV[-1] = thV[-1] - th_eps
        else:
            # linear-scale search
            lb = thV[max([0, imax-1])]
            ub = thV[min([find_N-1, imax+1])]
            thV = np.linspace(lb, ub, find_N)
            
        for iTH in range(find_N):
            Sth = O > thV[iTH]
            for i in range(N):
                x, y, s = X[i,:], Y[i,:], Sth[i,:]
                xi = xi_estimation(y, s, B)
                rsnr[i] = RSNR(x, np.dot(D, xi))
            if FoM=='arsnr':
                fom_v[iTH] = np.mean(rsnr)
            elif FoM=='pcr':
                fom_v[iTH] = np.sum(rsnr>RSNRmin)/N
            else:
                raise ValueError('arsnr and pcr are acceptable FoMs')
        imax = np.argmax(fom_v)
        th = thV[imax]
        maxfom = fom_v[imax]
        if verbose:
            print(' => {:d}) ibest = {:d}, th = {:.6e}  ({} = {:.2f})'.format(iL, imax, th, FoM ,maxfom))
    return th, maxfom

def decoder_AE(X, D, model, th_onn, A=None, isnr=100, th_xi=0):
    
    '''
    input:
    X     : input vectors, ndarray
    D     : Sparsity Basis, ndarray
    model : DNN model
    th_onn: threshold for model prediction, float
    A     : matrix representing the ENC-layer, ndarray
    isnr  : level of additive white gaussian noise. float
    th_xi : threshold for xi_estimation, float
    
    output:
    ret: Dictionary with model prediction, support, reconstructed signal, rsnr and rmnr
    '''
    
    if A is None:
        A = model.layers[1].get_weights()[0].T
        l_type = getattr(model.get_layer(index=0),'type_l','Base')
        if l_type=='Antipodal':
            A = np.sign(A)
    
    B = A@D
    Xn = add_awgn(X, SNRdB=isnr)
    Y = np.dot(A, Xn.T).T
    print('predict ...', end=' ')
    Onn = model.predict(Xn)
    Supp = (Onn > th_onn)

    print('evaluate ...', end='\r')
    print('\r')
    Xi = np.zeros(X.shape)
    for i, (y, s) in enumerate(zip(Y, Supp)):
        Xi[i,:] = xi_estimation(y, s, B)

    Xr = np.dot(D, Xi.T).T
    Yp = np.dot(B, Xi.T).T
    Supp = (np.abs(Xi) > th_xi)
    rmnr = RSNR(Y, Yp)
    rsnr = RSNR(X, Xr)
    
    ret = {'Onn': Onn, 'Supp': Supp, 'Xr': Xr, 'rsnr': rsnr, 'rmnr': rmnr}
    return ret

def decoder(X, D, model, th_onn, A, isnr=100, th_xi=0):
    
    '''
    input:
    X     : input vectors, ndarray
    D     : Sparsity Basis, ndarray
    model : DNN model
    th_onn: threshold for model prediction, float
    A     : matrix representing the ENC-layer, ndarray
    isnr  : level of additive white gaussian noise. float
    th_xi : threshold for xi_estimation, float
    
    output:
    ret: Dictionary with model prediction, support, reconstructed signal, rsnr and rmnr
    '''
    
    B = A@D
    Xn = add_awgn(X, SNRdB=isnr)
    Y = np.dot(A, Xn.T).T
    print('predict ...', end=' ')
    Onn = model.predict(Y)
    Supp = (Onn > th_onn)

    print('evaluate ...', end='\r')
    Xi = np.zeros(X.shape)
    for i, (y, s) in enumerate(zip(Y, Supp)):
        Xi[i,:] = xi_estimation(y, s, B)

    Xr = np.dot(D, Xi.T).T
    Yp = np.dot(B, Xi.T).T
    Supp = (np.abs(Xi) > th_xi)
    rmnr = RSNR(Y, Yp)
    rsnr = RSNR(X, Xr)
    
    ret = {'Onn': Onn, 'Supp': Supp, 'Xr': Xr, 'rsnr': rsnr, 'rmnr': rmnr}
    return ret


def find_Fidelity(df, a, m, enc, isnr, Sx, kappa):
   
    '''
    Found useless parameter to define the fidelity of the reconstructed signal
    
    input:
    df    : data frame to be used
    a     : alpha parameters
    m     : number of element of y
    enc   : encoder type
    isnr  : value of additive white gaussian noise
    Sx    : data to decoder casted to boolean
    kappa : sparsity value
    
    output:
    rmnr   : Reconstruction Measurement to Noise Ratio
    rsnr   : Reconstruction Signal to Noise Ratio
    Supp   : Support solution
    se     : succeeded element
    nse    : negative succeeded element
    pse    : positive succeeded element
    labels :
    '''
    
    rmnr = df['rmnr'].xs([a, m, enc, isnr], level=['alpha','m', 'enc', 'ISNR']).values[0]
    rsnr = df['rsnr'].xs([a, m, enc, isnr], level=['alpha','m', 'enc', 'ISNR']).values[0]
    Supp = df['Supp'].xs([a, m, enc, isnr], level=['alpha','m', 'enc', 'ISNR']).values[0]
            
    tp = np.array([np.sum( s[ y]) for s, y in zip(Supp, Sx)])
    labels = (tp >= kappa)
            
    se = np.sum((rmnr > 150) & (rsnr < 150))
    nse = np.sum((rmnr > 150) & (rsnr < 150) & ~labels)
    pse = np.sum((rmnr > 150) & (rsnr < 150) &  labels)
    print('ISNR={}) bad corner for {}: {} elements, of which {} positive and {} negative'.format(isnr,enc,se,pse,nse))
    
    return rmnr, rsnr, Supp, se, nse, pse, labels




def reconstruct_batch(D, A, B, s_batch, y_batch, th):
    n = D.get_dimension()
    x_hat_batch = np.zeros(shape=(len(s_batch), n))
    for j, s in enumerate(s_batch):
        Bk = B[:, s >= th] # Select the k columns which bring information
        Bkpinv = np.linalg.pinv(Bk) # Evaluate the pseudoinverse matrix
        # Evaluate reconstructed signal
        csi_hat_k = np.dot(Bkpinv, y_batch[j])
        csi_hat = np.zeros(shape = n)
        csi_hat[s >= th] = csi_hat_k
        x_hat_batch[j,:] = np.dot(D.get_matrix(), csi_hat)
        # Progress bar
        prog = 100*(j+1)//len(s_batch)
        if (j+1) % (len(s_batch)//10) == 0:
            if type(th) != np.float64:
                sys.stdout.write('\r    Reconstructing -> {:3d}% '.format(prog)+'#'*(prog//10)+'.'*(10-(prog//10))+' ')
            else:
                sys.stdout.write('\r    Reconstructing (th = {:.4f}) -> {:3d}% '.format(th, prog)+'#'*(prog//10)+'.'*(10-(prog//10))+' ')
            sys.stdout.flush()
    return x_hat_batch

def evaluate_reconstruction(x_batch, x_hat_batch, rsnrmin, get_rsnr_list = False):
    eps = np.finfo(float).eps # avoid division by 0
    # rsnr list
    rsnr = (20*np.log10(np.linalg.norm(x_hat_batch, 2, 1)/(np.linalg.norm(x_hat_batch-x_batch, 2, 1)+eps)+eps))
    # arsnr evaluation
    arsnr = (sum(rsnr)/len(rsnr))
    # pcr evaluation
    pcr = (sum(rsnr >= rsnrmin)/len(rsnr))
    
    if get_rsnr_list:
        return rsnr, arsnr, pcr
    
    return arsnr, pcr

