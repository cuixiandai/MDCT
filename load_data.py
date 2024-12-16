import numpy as np
from spectral.io import envi
import os.path
from pathlib import Path
import scipy.io as sio


def load_data(name):
    if name == 'FL_T':
        path = 'Datasets/Flevoland/T_Flevoland_14cls.mat'
        first_read =sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)
        T[: ,:, 0]=first_read
        del first_read
        T[: ,:, 1]=sio.loadmat(path)['T22']
        T[: ,:, 2]=sio.loadmat(path)['T33']
        T[: ,:, 3]=sio.loadmat(path)['T12']
        T[: ,:, 4]=sio.loadmat(path)['T13']
        T[: ,:, 5]=sio.loadmat(path)['T23']
        labels = sio.loadmat('Datasets/Flevoland/Flevoland_gt.mat')['gt'] 
##############################################################################

    elif name == 'SF':
        first_read = sio.loadmat('Datasets/san_francisco/SanFrancisco_Coh.mat')['T']
        first_read=first_read.astype(np.complex64)
        T = np.zeros(first_read.shape[:2] + (6,), dtype=np.complex64)
        T[: ,:, 0]=first_read[: ,:, 0]
        T[: ,:, 1]=first_read[: ,:, 3]
        T[: ,:, 2]=first_read[: ,:, 5]  
        T[: ,:, 3]=first_read[: ,:, 1]
        T[: ,:, 4]=first_read[: ,:, 2] 
        T[: ,:, 5]=first_read[: ,:, 4] 
        del first_read 
        labels = sio.loadmat('Datasets/san_francisco/SanFrancisco_gt.mat')['gt'] 

##############################################################################
    elif name == 'ober':
        path = 'Datasets/Oberpfaffenhofen/T_Germany.mat'
        first_read =sio.loadmat(path)['T11']
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)
        T[: ,:, 0]=first_read
        del first_read
        T[: ,:, 1]=sio.loadmat(path)['T22']
        T[: ,:, 2]=sio.loadmat(path)['T33']
        T[: ,:, 3]=sio.loadmat(path)['T12']
        T[: ,:, 4]=sio.loadmat(path)['T13']
        T[: ,:, 5]=sio.loadmat(path)['T23']

        labels = sio.loadmat('Datasets/Oberpfaffenhofen/Label_Germany.mat')['label']

##############################################################################
    else:
        print("Incorrect data name")
        
    return T, labels

