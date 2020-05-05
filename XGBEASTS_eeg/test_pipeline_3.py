#from eeg_learn_functions import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from numpy import genfromtxt
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m
import pickle
import scipy.io
from tensorflow.keras.metrics import AUC
#import theano
#import theano.tensor as T

from scipy.interpolate import griddata
from sklearn.preprocessing import scale


from sklearn.preprocessing import StandardScaler
import keras
theta = (4,8)
alpha = (8,12)
beta = (12,40)


class TestPipelineEEG:
    
    def __init__(self, image_size = 28, frame_duration = 0.78, overlap = 0.0, model_path = 'path_to_my_model.h5', normalize = True):
        self.locs_2d = [(-2.0,4.0),
                   (2.0,4.0),
                   (-1.0,3.0),
                   (1.0,3.0),
                   (-3.0,3.0),
                   (3.0,3.0),
                   (-2.0,2.0),
                   (2.0,2.0),
                   (-2.0,-2.0),
                   (2.0,-2.0),
                   (-4.0,1.0),
                   (4.0,1.0),
                   (-1.0,-3.0),
                   (1.0,-3.0)]        
        self.image_size = image_size
        self.frame_duration = frame_duration
        self.overlap = overlap
        self.model = keras.models.load_model(model_path)
        self.normalize = normalize
    
    def get_fft(self, snippet):
        Fs = 128.0;  # sampling rate
        #Ts = len(snippet)/Fs/Fs; # sampling interval
        snippet_time = len(snippet)/Fs
        Ts = 1.0/Fs; # sampling interval
        t = np.arange(0,snippet_time,Ts) # time vector

        # ff = 5;   # frequency of the signal
        # y = np.sin(2*np.pi*ff*t)
        y = snippet
    #     print('Ts: ',Ts)
    #     print(t)
    #     print(y.shape)
        n = len(y) # length of the signal
        k = np.arange(n)
        T = n/Fs
        frq = k/T # two sides frequency range
        frq = frq[range(n//2)] # one side frequency range

        Y = np.fft.fft(y)/n # fft computing and normalization
        Y = Y[range(n//2)]
        #Added in: (To remove bias.)
        #Y[0] = 0
        return frq,abs(Y)

    def theta_alpha_beta_averages(self, f,Y):
        theta_range = (4,8)
        alpha_range = (8,12)
        beta_range = (12,40)
        theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
        alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
        beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
        return theta, alpha, beta


    def make_steps(self, samples,frame_duration,overlap):
        '''
        in:
        samples - number of samples in the session
        frame_duration - frame duration in seconds 
        overlap - float fraction of frame to overlap in range (0,1)

        out: list of tuple ranges
        '''
        #steps = np.arange(0,len(df),frame_length)
        Fs = 128
        i = 0
        intervals = []
        samples_per_frame = 100 #Fs * frame_duration
        while i+samples_per_frame <= samples:
            intervals.append((i,i+samples_per_frame))
            i = i + samples_per_frame - int(samples_per_frame*overlap)
        return intervals

    def make_frames(self, df,frame_duration):
        '''
        in: dataframe or array with all channels, frame duration in seconds
        out: array of theta, alpha, beta averages for each probe for each time step
            shape: (n-frames,m-probes,k-brainwave bands)
        '''
        Fs = 128.0
        frame_length = 100#Fs*frame_duration
        frames = []
        #steps = make_steps(len(df),frame_duration,overlap)

        frame = []
        for channel in df.columns:
            snippet = np.array(df.loc[:,int(channel)])
            f,Y =  self.get_fft(snippet)
            theta, alpha, beta = self.theta_alpha_beta_averages(f,Y)
            frame.append([theta, alpha, beta])
        frames.append(frame)
        return np.array(frames)


    def make_data_pipeline(self, df,image_size, frame_duration):
        '''
        IN: 
        file_names - list of strings for each input file (one for each subject)
        labels - list of labels for each
        image_size - int size of output images in form (x, x)
        frame_duration - time length of each frame (seconds)
        overlap - float fraction of frame to overlap in range (0,1)

        OUT:
        X: np array of frames (unshuffled)
        y: np array of label for each frame (1 or 0)
        '''
        ##################################
        ###Still need to do the overlap###!!!
        ##################################

        Fs = 128.0   #sampling rate
        frame_length = 100#Fs * frame_duration



        X_0 = self.make_frames(df,frame_duration)
        #steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0),14*3)

        images = self.gen_images(np.array(self.locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3) 
        X = images

        if(self.normalize):
            #with open('scalers_dump.pickle', 'rb') as f:
            #    scalers = pickle.load(f)
            #X_r = scalers[0].transform(X[:,:,:,0].reshape((X.shape[0]*image_size, image_size)))
            #X_g = scalers[0].transform(X[:,:,:,1].reshape((X.shape[0]*image_size, image_size)))
            #X_b = scalers[0].transform(X[:,:,:,2].reshape((X.shape[0]*image_size, image_size)))
            #X_x = scalers[0].transform(X[:,:,:,3].reshape((X.shape[0]*image_size, image_size)))
            #X_y = scalers[0].transform(X[:,:,:,4].reshape((X.shape[0]*image_size, image_size)))
            ##
            #X[:,:,:,0] = X_r.reshape((X.shape[0], X.shape[1], X.shape[2])) 
            #X[:,:,:,1] = X_g.reshape((X.shape[0], X.shape[1], X.shape[2])) 
            #X[:,:,:,2] = X_b.reshape((X.shape[0], X.shape[1], X.shape[2]))   
            #X[:,:,:,3] = X_x.reshape((X.shape[0], X.shape[1], X.shape[2])) 
            #X[:,:,:,4] = X_y.reshape((X.shape[0], X.shape[1], X.shape[2]))
            X_r = X[:,:,:,0].reshape((X.shape[0]*image_size, image_size))
            X_g = X[:,:,:,1].reshape((X.shape[0]*image_size, image_size))
            X_b = X[:,:,:,2].reshape((X.shape[0]*image_size, image_size))
    
            X[:,:,:,0] = scale(X_r, axis = 1).reshape((X.shape[0], X.shape[1], X.shape[2])) 
            X[:,:,:,1] = scale(X_g, axis = 1).reshape((X.shape[0], X.shape[1], X.shape[2])) 
            X[:,:,:,2] = scale(X_b, axis = 1).reshape((X.shape[0], X.shape[1], X.shape[2])) 
        return X


    def gen_images(self, locs, features, n_gridpoints, normalize=True,
                   augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False, train = True, scaler_list_dump = []):
        """
        Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

        :param locs: An array with shape [n_electrodes, 2] containing X, Y
                            coordinates for each electrode.
        :param features: Feature matrix as [n_samples, n_features]
                                    Features are as columns.
                                    Features corresponding to each frequency band are concatenated.
                                    (alpha1, alpha2, ..., beta1, beta2,...)
        :param n_gridpoints: Number of pixels in the output images
        :param normalize:   Flag for whether to normalize each band over all samples
        :param augment:     Flag for generating augmented images
        :param pca:         Flag for PCA based data augmentation
        :param std_mult     Multiplier for std of added noise
        :param n_components: Number of components in PCA to retain for augmentation
        :param edgeless:    If True generates edgeless images by adding artificial channels
                            at four corners of the image with value = 0 (default=False).
        :return:            Tensor of size [samples, colors, W, H] containing generated
                            images.
        """
        feat_array_temp = []
        nElectrodes = locs.shape[0]     # Number of electrodes
        # Test whether the feature vector length is divisible by number of electrodes
        assert features.shape[1] % nElectrodes == 0
        n_colors = features.shape[1] // nElectrodes
        for c in range(int(n_colors)):
            feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
        if augment:
            if pca:
                for c in range(n_colors):
                    feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
            else:
                for c in range(n_colors):
                    feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
        nSamples = features.shape[0]
        # Interpolate the values
        grid_x, grid_y = np.mgrid[
                         min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                         min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                         ]
        temp_interp = []
        for c in range(n_colors):
            temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
        # Generate edgeless images
        if edgeless:
            min_x, min_y = np.min(locs, axis=0)
            max_x, max_y = np.max(locs, axis=0)
            locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
            for c in range(n_colors):
                feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
        # Interpolating
        for i in range(nSamples):
            for c in range(n_colors):
                temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                        method='cubic', fill_value=np.nan)
            #print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
        # Normalizing
        scaler_list = []
        for c in range(n_colors):
            if normalize:
                if(train):
                    t = StandardScaler(with_mean = 0.5, with_std = 0.5).fit(temp_interp[c][~np.isnan(temp_interp[c])].reshape(-1, 1))
                    scaler_list.append(t)
                    temp_interp[c][~np.isnan(temp_interp[c])] = \
                        t.transform(temp_interp[c][~np.isnan(temp_interp[c])].reshape(-1, 1)).reshape(-1)
                else:
                    t = scaler_list_dump[c]
                    temp_interp[c][~np.isnan(temp_interp[c])] = \
                        t.transform(temp_interp[c][~np.isnan(temp_interp[c])].reshape(-1, 1)).reshape(-1)
            temp_interp[c] = np.nan_to_num(temp_interp[c])
        return np.swapaxes(np.asarray(temp_interp), 0, 1)   # swap axes to have [samples, colors, W, H]

    
    #def evaluate(self, df):
    #    X = self.make_data_pipeline(df,self.image_size, self.frame_duration)
    #    X = X.astype('float32')
    #    return self.model.predict(X).argmax(axis = -1)[0]
    def evaluate(self, df, threshold = 0.5):
        X = self.make_data_pipeline(df,self.image_size, self.frame_duration)
        X = X.astype('float32')
        res = self.model.predict(X)[:,0]<threshold
#res.astype('int')
        return res.astype('int')[0]
