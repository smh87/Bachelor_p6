from numpy.core.numeric import Infinity
import scipy.stats as stats
import os
import numpy as np
import math
import random
from collections import Counter
from dotenv import load_dotenv

def segment_data(scaled_X, frame_sizes, test_indices, sample_frequency, n_features):
    load_dotenv()
    Fs = sample_frequency*30 # 
    features = n_features

    def get_frames(df, frame_sizes):
        '''test pydoc'''

        
        frames = []
        labels = []
        used_frames = 0
        

        for i in range(len(frame_sizes)):
            frame_size = int(frame_sizes[i])
            input_vector = []
            for j in range(features):
                input_vector.append(np.resize(df[('feature'+str(j))].values[used_frames: used_frames + frame_size],(Fs)))
            
            # Retrieve the most often used label in this segment
            label = stats.mode(df['label'][used_frames: used_frames + frame_size])[0][0]
            frames.append(input_vector)
            labels.append(label)
            used_frames += frame_size


        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, Fs, features)
        labels = np.asarray(labels)

        return frames, labels

    X = []
    y = []    
    for i, frame in enumerate(scaled_X):
        
        sub_X, sub_y = get_frames(frame, frame_sizes[i])
        sub_X = sub_X.reshape(sub_X.shape[0], sub_X.shape[1], features, 1)
        X.append(sub_X)
        y.append(sub_y)
        
    train_indices = set(range(len(scaled_X))) - set(test_indices)
    first_train = True
    for i in train_indices:
        if not first_train:
            X_train = np.concatenate((X_train, X[i]))
            y_train = np.concatenate((y_train, y[i]))
        else:
            X_train = X[i]
            y_train = y[i]
            first_train = False
    first_test = True
    for i in test_indices:
        if not first_test:
            X_test = np.concatenate((X_test, X[i]))
            y_test = np.concatenate((y_test, y[i]))
        else:
            X_test = X[i]
            y_test = y[i]
            first_test = False


    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
 
    return X_train, X_test, y_train, y_test
    # split data into 20 % test and 80 % train data
    #random_state controls the shuffling applied to the data before applying the split. stratify = y splits the data in a stratified fashion, using y as the class labels
  


