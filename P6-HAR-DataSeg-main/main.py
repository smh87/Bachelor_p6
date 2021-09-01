import preprocessing as pp
import PreProcessIMSHA as pp2
import preprocess_sport as pp3
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import CNN
import sys
import os
import gc
import Hierachical_nn as Hnn
from dotenv import load_dotenv

load_dotenv()
def slidingWindows(X_train, X_test, y_train, y_test, label, activity_channels):
  
    cnn = CNN.Cnn(activity_channels, X_train)
    result, confidence = cnn.run_cnn(X_train, X_test, y_train, y_test, label, 0)
    print('results are ', result)
    print('with confidence ', confidence)
    results = result[len(y_train):]
    mat = confusion_matrix(y_test, results)
    plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))
    F1 = f1_score(y_test, results, average='macro')
    accuracy = accuracy_score(y_test, results)
    recall = recall_score(y_test, results, average='macro')
    precision = precision_score(y_test, results, average='macro')

    print('accuracy: ', accuracy)
    plt.savefig('confusion_matrix_final.png')
    plt.close('all')
    gc.collect()
    return F1, accuracy, precision, recall, y_test, results

