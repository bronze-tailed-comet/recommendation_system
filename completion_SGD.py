
# coding: utf-8

# In[ ]:


from __future__ import (absolute_import, division, print_function,unicode_literals)
import os
import sys
import numpy as np
from tempfile import TemporaryFile
import pickle
import scipy.io
import numbers
from scipy.sparse import csr_matrix, find
from collections import defaultdict
import time

x = scipy.io.loadmat('netflix_data_app.mat')
x = x['netflix_data_app']

xt = scipy.io.loadmat('netflix_data_probe.mat')
xt = xt['netflix_data_probe']

rng = np.random.RandomState(42)
lr = .005
reg = .02
bias = True
init_mean = 0
init_std_dev = .1
k = 20
n_users = x.shape[0]
n_items = x.shape[1]

center = False

if(center):

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

else:
    
    n = np.sum(x>.5)
    
    if not bias:
        moy = 0
    else:
        moy = np.sum(x) / n
        print("Global average =",round(moy,3))
        
pu = rng.normal( init_mean , init_std_dev, ( n_users , k ) )
qi = rng.normal( init_mean , init_std_dev, ( n_items , k ) )
bu = np.zeros( n_users, np.double )
bi = np.zeros( n_items, np.double )

qs = find(x)
qs_u = qs[0]
qs_i = qs[1]
qs_r = qs[2]
lenqs = len(qs_u)

if(center):
    qst = find(xtn)
else:
    qst = find(xt)
    
qst_u = qst[0]
qst_i = qst[1]
qst_r = qst[2]
lenqst = len(qst_u)

print("Starting SVD...")

epoch = 0

choice = 'yes'

while( choice == 'yes' ):

    epoch+=1

    print("Epoch n°{}".format(epoch))
    
    for i in range(lenqs):

        b = i/lenqs*100
        sys.stdout.write("\r%f" % b + '%' )
        sys.stdout.flush()

        # compute current error
        dot = 0  

        for f in range(k):

            dot += qi[ qs_i[i] , f ] * pu[ qs_u[i] , f ]

        err = qs_r[i] - ( moy + bu[ qs_u[i] ] + bi[ qs_i[i] ] + dot )

        # update biases
        bu[ qs_u[i] ] += lr * ( err - reg * bu[ qs_u[i] ] )
        bi[ qs_i[i] ] += lr * ( err - reg * bi[ qs_i[i] ] )

        # update factors
        for f in range( k ) :

            puf = pu[ qs_u[i] , f ]
            qif = qi[ qs_i[i] , f ]
            pu[ qs_u[i] , f ] += lr * ( err * qif - reg * puf )
            qi[ qs_i[i] , f ] += lr * ( err * puf - reg * qif )

    print('Computing RMSE...')

    errg = 0

    for i in range(lenqst):

        est = moy + bu[ qst_u[i] ] + bi[ qst_i[i] ] + np.dot( pu[ qst_u[i] ] , qi[ qst_i[i] ] )
        
        err = (qst_r[i] - est)**2
        
        errg += err
        
    rmse = np.sqrt(errg / lenqst)

    print('RMSE :',rmse)
    print(' ')
    choice = input('Continue?').lower()

