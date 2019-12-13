import numpy as np
from scipy import linalg
from scipy import random

def add_awgn(x, SNRdB=100):
    '''
    add additive withe gaussian noise with a given SNR (dB)
    INPUT
    x: ndarray (TODO check sulle istanze per righe o per colonne)

    OUTPUT
    xn: copy of x corrupted by awgn
    '''
    if SNRdB < np.inf:
        ps = linalg.norm(x, axis=-1)
        noise = random.randn(*x.shape)
        pn0 = linalg.norm(noise, axis=-1)
        pn = ps/pn0/(10**(SNRdB/20))
        if len(x.shape) > 1:
            pn = np.tile(pn, (x.shape[1], 1)).T
        xn = x + noise * pn
    else:
        xn = x
    return xn

def RSNR (x, xr):
    return 10*np.log10(np.sum(np.array(x)**2, axis=-1)/np.sum((np.array(x) - np.array(xr))**2, axis=-1))

def xi_estimation(y, s, B):
    '''
    input:
    B: matrix (m, n) np.dot(A, D)
    y: vector (m,) measurement vector np.dot(A, x)
       or a matrix (N, m) where rows are measurement vectors
    s: support vector (n,)
    output:
    xi: sparsity vector (n,)
    '''
    n = B.shape[1]
    xik = np.dot(linalg.pinv(B[:, s.astype(bool)]), y.T)
    xi = np.zeros((n,))
    xi[s] = xik
    return xi

# signal pre-processing
def sparsify(x_batch, spa, basis):
    batch_size = x_batch.shape[0]
    n = x_batch.shape[1]
    x_sparse = np.zeros(shape=[batch_size, n])
    support = np.zeros(shape=[batch_size, n])
    for i in range(batch_size):
        csi = np.dot(basis.get_matrix(True), x_batch[i])
        csi_sort = np.argsort(np.abs(csi))
        csi_sparse = np.zeros(shape=n)
        for index in csi_sort[-spa:]:
            if(np.abs(csi[index]) > 0):
                csi_sparse[index] = csi[index]
                support[i][index] = 1
        x_sparse[i] = np.dot(basis.get_matrix(), csi_sparse)
    return x_sparse, support


def Quantizer(A, size, level, epsilon):
    '''
    The function evaluate optimum value for quantization
    input:
    A       : Matrix. ndarray
    size    : Dimension of the Matrix. List
    level   : Number of level in the quantization. Scalar
    epsilon : Target error. Scalar
    
    output:
    a       : Level value. ndarray
    Aq      : Quantized matrix. ndarray
    '''
    # Creo un vettore con i valori della matrice
    Avec = np.asarray(A).reshape(-1)
    Avecsort = np.sort(Avec)
    Aq = A.copy()
    
    # Inizializzo il vettore dei valori quantizzati e delle soglie
    a = np.linspace(start = Avecsort.min(), stop = Avecsort.max(), num = level, endpoint = True)
    b = np.zeros(level+1,)
    b[0] = Avecsort.min()
    b[level] = Avecsort.max()
    print('b = {}'.format(b))    
    print('a = {}'.format(a))
    
    diff = 100*np.ones(size[0]*size[1],)
    diffprev = np.zeros(size[0]*size[1],)
    aprev = np.zeros(level,)
    iterval = 0
    
    while diff.all() >= epsilon:
        # Calcolo soglie ottime
        for i in range(0,level-1):
            b[i+1] = (a[i]+a[i+1])/2
        print('b = {}'.format(b))
        # Calcolo valori di a ottimi
        for i in range(0, level):
            number = 0
            counter = 0
            for val in Avecsort:
                if (val > b[i]) and (val<=b[i+1]):
                    number = number + val
                    counter = counter + 1
            if counter != 0:
                a[i] = number/counter
            else:
                a[i] = (b[i]+b[i+1])/2
        print('a = {}'.format(a))
        
        # Calcolo metrica
        iterval = 0
        diff = np.zeros(size[0]*size[1],)
        for idx,i in enumerate(Avecsort):
            for j in range(level):
                if (i>=b[j]) and (i<=b[j+1]):
                    diff[idx] = (abs(i) - abs(a[j]))**2
                    iterval = iterval + 1
        
        check = np.greater(diff,diffprev)
        if check.all():
            counter = np.greater(diff,epsilon*np.ones(size[0]*size[1],))
            counter = sum(counter)
            print('The given epsilon is reached by {} element'.format(size[0]*size[1]-counter))
            break
        else:
            diffprev = diff
            aprev = a.copy()

    # Quantizzazione della matrice
    for r in range(size[0]):
        for c in range(size[1]):
            for idx in range(level):
                if (A[r,c]>b[idx])and(A[r,c]<b[idx+1]):
                    Aq[r,c] = a[idx]
    return aprev, Aq