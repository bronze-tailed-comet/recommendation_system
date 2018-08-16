
# coding: utf-8

# In[ ]:


import os
import sys
import scipy.io
import numpy as np
from scipy.sparse import find , csc_matrix , csr_matrix
from scipy.sparse.linalg import svds, eigs

def ridge_reg( target_matrix , fixed_matrix , data , lamb ):

    for j in range(target_matrix.shape[1]):
        
        b = j/target_matrix.shape[1]*100
        sys.stdout.write("\r%f" % b + '%' )
        sys.stdout.flush()

        nonzeros = data[:,j].nonzero()[0]
        y = data[nonzeros, j].todense()
        X = fixed_matrix[:, nonzeros].T

        target_matrix[:,j] = np.squeeze(np.linalg.inv(X.T.dot(X) + lamb * np.eye(X.shape[1])).dot(X.T.dot(y)))



def rmse( f , U , DV ):

    errg = 0

    for i in range(len(f[1])):

        est = np.dot( U[ f[0][i] , : ] , DV[ : , f[1][i] ] )

        err = ( f[2][i] - est)**2

        errg += err

    rmse_ = np.sqrt( errg / len(f[1]) )

    return rmse_



x = scipy.io.loadmat('netflix_data_app.mat')
x = x['netflix_data_app']

xt = scipy.io.loadmat('netflix_data_probe.mat')
xt = xt['netflix_data_probe']

n = np.sum(x>.5)
nt = np.sum(xt>.5)
moy = np.sum(x) / n

Mask = x>.5
Maskt = xt>.5

s = moy * Mask
x = x - s
xtn = xt - Maskt.multiply(moy)

mf = x.sum(axis=1) / Mask.sum(axis=1)

s = Mask.multiply(mf)

x = x - s
xtn = xtn - Maskt.multiply(mf)

ms = x.sum(axis=0) / Mask.sum(axis=0)

s = Mask.multiply(ms)

x = x - s
xtn = xtn - Maskt.multiply(ms)

f = find(xtn)

k = 6

U , D , Vt = svds( x , k )

U = U[ : , ::-1 ]
D = D[ ::-1 ]
Vt = Vt[ ::-1 , : ]
DV = np.dot( np.diag(D) , Vt )



epoch = 0 # choose the number of epoch you want 

lamb = 2 # regularization

choice = 'yes'

while( choice == 'yes' ):
    
    epoch+=1

    print("Epoch n°{}".format(epoch))

    for ne in range(n_epoch):
    
        print('Epoch n°',ne+1)
        print('\nDV / U.T')
        print('\n')
        ridge_reg( DV , U.T , x , lamb )
        print('\nU.T / DV')
        ridge_reg( U.T , DV , x , lamb )
    
    print('Computing RMSE...')
    print('RMSE :',rmse(f,U,DV))
    print(' ')
    choice = input('Continue?').lower()


