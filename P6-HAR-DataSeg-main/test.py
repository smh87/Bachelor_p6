from scipy.sparse import data
from tensorflow.python.eager.def_function import FREQUENT_TRACING_WARNING_THRESHOLD
from tensorflow.python.keras.utils.generic_utils import skip_failed_serialization
import Data_Segmentation as ds
import Hiearachical_cross_validation as hcv
import preprocessing as pp
import PreProcessIMSHA as pp2
import Preprocess_Mhealth as pp3
import CNN
import gc
import Hierachical_nn as Hnn
import numpy as  np
import main
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def test(dataset, method, sample_frequency, activity_channels, n_features, threshold = 0):
  

    if dataset == 'WISDM':
        scaled_X, label = pp.preprocess()
    elif dataset == 'IMSHA':
        scaled_X, label = pp2.preprocess2()
    elif dataset == 'DAILY_AND_SPORT':
        scaled_X, label = pp3.preprocess()
    else:
        exit()
    F1_array = []
    accuracy_array = []
    recall_array = []
    precision_array = []
    all_y_test = np.array([])
    all_results = np.array([])
    result = .0
    if method == '1':
        # leave K-out cross validation
        # data segmentation
        split_data_X, split_data_y = ds.segment_data(scaled_X, sample_frequency, n_features)
        for i in range(len(split_data_y)):
            # Copy the data and labels to prevent mutating the original values
            X_test = np.copy(split_data_X[i])
            y_test = np.copy(split_data_y[i])
            second_test = -1
            first = 0
            # Get the set of all labels in the dataset
            all_labels = set(label.transform(label.classes_))
            # get the set of missing labels by comparing the difference between all labels and the labels in the test set
            missing_labels = all_labels.difference(set(y_test))
            # if labels are missing, find a test person with all of them and add them to the test set 
            if missing_labels:
                second_test = i
                while not missing_labels.issubset(set(split_data_y[second_test])):
                    print('are the missing labesl ', missing_labels, ' in the set ', set(split_data_y[second_test]))
                    second_test += 1
                    if second_test >= len(split_data_y):
                        second_test = 0
                X_test = np.concatenate((X_test, split_data_X[second_test]))
                y_test = np.concatenate((y_test, split_data_y[second_test]))
                if second_test + 1 < len(split_data_y):
                    first = second_test + 1
                else:
                    first = 0
            else:
                if i + 1 < len(split_data_y):
                    first = i + 1
                else:
                    first = 0
            
            
            
            X_train = np.copy(split_data_X[first])
            y_train = np.copy(split_data_y[first])
            for j in range(len(split_data_y)):
                if j != i and j != first and j != second_test:
                    X_train = np.concatenate((X_train, split_data_X[j])) 
                    y_train = np.concatenate((y_train, split_data_y[j])) 
            # Convert all data to numpy arrays
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

        # end leave K out cross validation
            F1, accuracy, precision, recall, returned_y, results = main.slidingWindows(X_train, X_test, y_train, y_test, label, activity_channels)
            if(all_y_test.size == 0):
                all_y_test = returned_y
                all_results = results
            else:
                all_y_test = np.concatenate((all_y_test, returned_y))
                all_results = np.concatenate((all_results, results))
            F1_array.append(F1)
            accuracy_array.append(accuracy)
            recall_array.append(recall)
            precision_array.append(precision)
            result += F1





    elif method == '2':
        
        frame_sizes = []
        for _ in scaled_X:
            frame_sizes.append([])
        for i in range(len(scaled_X)):
            for j in range(int(len(scaled_X[i]) / (sample_frequency* 30))):
                frame_sizes[i].append(sample_frequency*30) #create weird tree :)
        # leave K out cross validation
        split_data_X, split_data_y = hcv.segment_data(scaled_X.copy(), frame_sizes, sample_frequency, n_features)
        for i in range(len(split_data_y)):
            test_indices = []
            test_indices.append(i)
            X_test = np.copy(split_data_X[i])
            y_test = np.copy(split_data_y[i])
            second_test = -1
            first = 0
            all_labels = set(label.transform(label.classes_))
            missing_labels = all_labels.difference(set(y_test))
            if missing_labels:
                second_test = i
                while not missing_labels.issubset(set(split_data_y[second_test])):
                    second_test += 1
                    if second_test >= len(split_data_y):
                        second_test = 0
                X_test = np.concatenate((X_test, split_data_X[second_test]))
                y_test = np.concatenate((y_test, split_data_y[second_test]))
                test_indices.append(second_test)
                if second_test + 1 < len(split_data_y):
                    first = second_test + 1
                else:
                    first = 0
            else:
                if i + 1 < len(split_data_y):
                    first = i + 1
                else:
                    first = 0

            X_train = split_data_X[first]
            y_train = split_data_y[first]
            for j in range(len(split_data_y)):
                if j != i and j != first and j != second_test:
                    X_train = np.concatenate((X_train, split_data_X[j])) 
                    y_train = np.concatenate((y_train, split_data_y[j]))
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
        # end leave K out cross validation
            F1, accuracy, precision, recall, returned_y, results = Hnn.hierachical(scaled_X.copy(), X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), label, frame_sizes.copy(), 0, test_indices, False, sample_frequency, activity_channels, threshold, n_features)
            if(all_y_test.size == 0):
                all_y_test = returned_y
                all_results = results
            else:
                all_y_test = np.concatenate((all_y_test, returned_y))
                all_results = np.concatenate((all_results, results))
            F1_array.append(F1)
            accuracy_array.append(accuracy)
            recall_array.append(recall)
            precision_array.append(precision)
            result += F1





    elif method == '3':
        frame_sizes = []
        for _ in scaled_X:
            frame_sizes.append([])
        for i in range(len(scaled_X)):
            for j in range(int(len(scaled_X[i]) / (sample_frequency* 30))):
                frame_sizes[i].append(sample_frequency*30) #create weird tree :)
        # leave k out cross validation
        split_data_X, split_data_y = hcv.segment_data(scaled_X.copy(), frame_sizes, sample_frequency, n_features)
        for i in range(len(split_data_y)):
            test_indices = []
            test_indices.append(i)
            X_test = np.copy(split_data_X[i])
            y_test = np.copy(split_data_y[i])
            second_test = -1
            first = 0
            all_labels = set(label.transform(label.classes_))
            missing_labels = all_labels.difference(set(y_test))
            if missing_labels:
                second_test = i
                while not missing_labels.issubset(set(split_data_y[second_test])):
                    second_test += 1
                    if second_test >= len(split_data_y):
                        second_test = 0
                X_test = np.concatenate((X_test, split_data_X[second_test]))
                y_test = np.concatenate((y_test, split_data_y[second_test]))
                test_indices.append(second_test)
                if second_test + 1 < len(split_data_y):
                    first = second_test + 1
                else:
                    first = 0
            else:
                if i + 1 < len(split_data_y):
                    first = i + 1
                else:
                    first = 0

            X_train = split_data_X[first]
            y_train = split_data_y[first]
            for j in range(len(split_data_y)):
                if j != i and j != first and j != second_test:
                    X_train = np.concatenate((X_train, split_data_X[j])) 
                    y_train = np.concatenate((y_train, split_data_y[j]))
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
        # end leave k out cross validation
            F1, accuracy, precision, recall, returned_y, results = Hnn.hierachical(scaled_X.copy(), X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), label, frame_sizes.copy(), 0, test_indices, True, sample_frequency, activity_channels, threshold, n_features)
            if(all_y_test.size == 0):
                all_y_test = returned_y
                all_results = results
            else:
                all_y_test = np.concatenate((all_y_test, returned_y))
                all_results = np.concatenate((all_results, results))
            F1_array.append(F1)
            accuracy_array.append(accuracy)
            recall_array.append(recall)
            precision_array.append(precision)
            result += F1
    else:
        print(method)
        print('input should be: python test [1 for sliding/ 2 for hierarchical / 3 for hirachical with baye split] [DATASET]')
        exit()
    total_F1 = 0
    total_accuracy = 0
    total_recall = 0
    total_precission = 0
    for score in F1_array:
        total_F1 += score
    for score in accuracy_array:
        total_accuracy += score
    for score in precision_array:
        total_precission += score
    for score in recall_array:
        total_recall += score
    print('All of the F1 results: ', F1_array)
    print('All accuracies: ', accuracy_array)
    print('All precissions: ', precision_array)
    print('All recall values: ', recall_array)
    with open('results.txt', 'a') as output:
        if method == '1':
            output.write(f'\nResults for Sliding windows with frequency {sample_frequency} for dataset {dataset}\n')
        elif method == '2':
            output.write(f'\nResults for Hierarchical with frequency {sample_frequency} for dataset {dataset}\n')
        else:
            output.write(f'\nResults for Hierarchical with bayesian split with frequency {sample_frequency} and threshold {threshold} for dataset {dataset}\n')
        mat = confusion_matrix(all_y_test, all_results)
        plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(20,20),colorbar=True)
        plt.savefig(f'confusion_matrix_freq{sample_frequency}_thresh{threshold}_dataset{dataset}.png')
        output.write(f'Avererage F1: {total_F1 / len(split_data_y)}\n')
        output.write(f'Avererage accuracy: {total_accuracy / len(split_data_y)}\n')
        output.write(f'Avererage recall: {total_recall / len(split_data_y)}\n')
        output.write(f'Avererage recall: {total_precission / len(split_data_y)}\n')

# Main for running all of our tests 
if __name__ == "__main__":
    all_frequencies = [20, 10, 5]
    all_thresholds = [0.5, 0.75, 0.9]
    all_datasets = [ 'IMSHA', 'WISDM', 'DAILY_AND_SPORT']
    all_methods = [ '1','2', '3']
    n_features = {'IMSHA':27, 'WISDM':3, 'DAILY_AND_SPORT':45}
    all_activity_channals = {'IMSHA':11, 'WISDM':6, 'DAILY_AND_SPORT':19}
    for frequency in all_frequencies:
        for method in all_methods:
            for dataset in all_datasets:
                if method == '3':
                    for threshold in all_thresholds:
                        test(dataset, method, frequency, all_activity_channals[dataset], n_features[dataset], threshold)
                        gc.collect()
                else:
                    test(dataset, method, frequency, all_activity_channals[dataset], n_features[dataset])
                    gc.collect()
