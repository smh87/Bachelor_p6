import scipy.stats as stats
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv

def segment_data(scaled_X, sample_frequency, n_features):
    load_dotenv()
    Fs = sample_frequency
    frame_size = Fs*4 
    hop_size = Fs*2 
    features = n_features

    def get_frames(df, frame_size, hop_size):

        

        frames = []
        labels = []
        for i in range(0, len(df) - frame_size, hop_size):
            input_vector = []
            for j in range(features):
                input_vector.append(df[('feature'+str(j))].values[i: i + frame_size])
            
            # Retrieve the most often used label in this segment
            label = stats.mode(df['label'][i: i + frame_size])[0][0]
            frames.append(input_vector)
            labels.append(label)

        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, features)
        labels = np.asarray(labels)

        return frames, labels


    

    # split data into 20 % test and 80 % train data
    #random_state controls the shuffling applied to the data before applying the split. stratify = y splits the data in a stratified fashion, using y as the class labels

    X = []
    y = []    
    for frame in scaled_X:
        sub_X, sub_y = get_frames(frame, frame_size, hop_size)
        sub_X = sub_X.reshape(sub_X.shape[0], sub_X.shape[1], features, 1)
        X.append(sub_X)
        y.append(sub_y)
        


    return X, y