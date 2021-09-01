
import os
import gc
from dotenv import load_dotenv
import CNN
import Hierachical_Seg as hier
from sklearn.preprocessing import LabelEncoder
import numpy as  np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import sdt.changepoint as sdt

load_dotenv()



# ew, global variable :'c
class hashtable:
    def __init__(self):
        self.values = set()
    def save_value(self, value):
        self.values.add(value)
    def is_already_in_table(self, value):
        print('is it already in the table? ', value in self.values)
        return value in self.values
table = hashtable()

#run hierachical data segmentation
def hierachical(scaled_X, X_train, X_test, y_train, y_test, label, frame_sizes, depth, test_indices, bayesian_split, sample_frequency, activity_cannels, threshold, n_features):
    # print('train labels: ', y_train)
    # print('test labels: ', y_test)
    # cnns = {}
    cnn = CNN.Cnn(activity_cannels, X_train)
    result, confidence = cnn.run_cnn(X_train, X_test, y_train, y_test, label, depth)
    del cnn
    X_values = np.concatenate((X_train, X_test))
    
    return hierarchical_train(scaled_X, X_values, frame_sizes, label, depth, result, confidence, test_indices, bayesian_split, sample_frequency, threshold, n_features)

# 
def hierarchical_train(scaled_X, X_values, frame_sizes, label, depth, result, confidence, test_indices, bayesian_split, sample_frequency, threshold, n_features):
    depth += 1
    labels = list(set(label.transform(label.classes_)))
    # if len(cnns) == 0: #TODO: muligvis fjern
    #     for i in range(len(labels) - 1):
    #         for j in range(i + 1, len(labels)):
    #             cnns[str(labels[i]) + str(labels[j])] = CNN.Cnn(2, X_train)
    
    print('results are ', result)
    print('with confidence ', confidence)
    skip = False
    changed = 0
    for j in range(len(result)):
        if skip:
            skip = False
            continue
        i = (len(result)-1) - j
        frame_index, index = get_frame_indices(frame_sizes, i, test_indices)
        if frame_sizes[frame_index][index] / sample_frequency<= 7.5:
            continue
        if confidence[i] < 0.60642069:
            if bayesian_split:
                frame_sizes, change = split_bayesian_change_point(frame_sizes, X_values[i], i, test_indices, threshold)
                if change : changed += 1
            else:
                frame_sizes = split(frame_sizes, i, test_indices)
                changed += 1
        elif(result[i] != result[i - 1]):
            if bayesian_split:
                frame_sizes, change1 = split_bayesian_change_point(frame_sizes, X_values[i], i, test_indices, threshold)
                frame_sizes, change2 = split_bayesian_change_point(frame_sizes, X_values[i - 1], i - 1, test_indices, threshold)
                if change1 or change2 : changed += 1
                skip = True
            else:
                frame_sizes = split(frame_sizes, i, test_indices)
                frame_sizes = split(frame_sizes, i - 1, test_indices)
                changed += 1
                skip = True

   
    X_train, X_test, y_train, y_test  = hier.segment_data(scaled_X, frame_sizes, test_indices, sample_frequency, n_features)
   
    second_results = [[], []]

    for i in np.concatenate((X_train, X_test)):
        # the touples are in the form (result, probability)
        second_results[0].append(-1)
        second_results[1].append(-1)
    for i in range(len(labels)-1):
        for j in range(i + 1, len(labels)):
            temp_X_train = []
            temp_X_test = []
            temp_y_train = []
            temp_y_test = []
            indices = []
            print('labels should be ', labels[i], ' and ', labels[j])
            for k in range(len(np.concatenate((X_train, X_test)))):
                if k < len(X_train):
                    if labels[i] == y_train[k] or labels[j] == y_train[k]:
                        indices.append(k)
                        temp_X_train.append(X_train[k])
                        temp_y_train.append(y_train[k])
                else:
                    k -= len(X_train)
                    if labels[i] == y_test[k] or labels[j] == y_test[k]:
                        indices.append(k+len(X_train))
                        temp_X_test.append(X_test[k])
                        temp_y_test.append(y_test[k])
            temp_y_test = np.asarray(temp_y_test)
            temp_X_test = np.asarray(temp_X_test)
            temp_y_train = np.asarray(temp_y_train)
            temp_X_train = np.asarray(temp_X_train)
            small_label = LabelEncoder()
            label_map = [[labels[i], labels[j]]]
            label_map.append(small_label.fit_transform([label.classes_[labels[i]], label.classes_[labels[j]]]))
            for k in range(len(temp_y_train)):
                if temp_y_train[k] == label_map[0][0]:
                    temp_y_train[k] = label_map[1][0]
                else:
                    temp_y_train[k] = label_map[1][1]
            for k in range(len(temp_y_test)):
                if temp_y_test[k] == label_map[0][0]:
                    temp_y_test[k] = label_map[1][0]
                else:
                    temp_y_test[k] = label_map[1][1]
            small_cnn = CNN.Cnn(2, X_train)
            result, probability = small_cnn.run_cnn(temp_X_train, temp_X_test, temp_y_train, temp_y_test, small_label, depth)
            del small_cnn
            for k in range(len(indices)):
                if probability[k] >= second_results[1][indices[k]]:
                    second_results[0][indices[k]] = label_map[0][0] if result[k] == label_map[1][0] else label_map[0][1]
                    second_results[1][indices[k]] = probability[k]
    if changed > 20:
        return hierarchical_train(scaled_X, np.concatenate((X_train, X_test)), frame_sizes, label, depth, second_results[0], second_results[1], test_indices, bayesian_split, sample_frequency, threshold, n_features)
    else:
        print(second_results)
        print(y_test)
        results = second_results[0][len(y_train):]
        print(results)
        #mat = confusion_matrix(y_test, results)
        #plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))
        
        F1 = f1_score(y_test, results, average='macro')
        accuracy = accuracy_score(y_test, results)
        recall = recall_score(y_test, results, average='macro')
        precision = precision_score(y_test, results, average='macro')
        print('accuracy: ', accuracy)
        
        plt.close('all')
        gc.collect()
        return F1, accuracy, precision, recall, y_test, results

#split af window function 
def split(frame_sizes, index, test_indices):
    i, true_index = get_frame_indices(frame_sizes, index, test_indices)
    j = len(frame_sizes[i])-1

    frame_sizes[i].append(frame_sizes[i][j])
    while(j > true_index):
        j -= 1
        frame_sizes[i][j+1] = frame_sizes[i][j]

    frame_sizes[i][j+1] = frame_sizes[i][j] / 2
    frame_sizes[i][j] = frame_sizes[i][j] / 2
    return frame_sizes

def get_frame_indices(frame_sizes, index, test_indices):
    i = 0
    j = 0
    while i in test_indices:
        i += 1 
    true_index = index
    passed_train = False
    while not passed_train and len(frame_sizes[i]) <= true_index:
        true_index -= len(frame_sizes[i])
        i += 1
        while i in test_indices:
            i += 1
        if i >= len(frame_sizes):
            passed_train = True
    if passed_train:
        i = test_indices[j]
    while passed_train and len(frame_sizes[i]) <= true_index:
        true_index -= len(frame_sizes[i])
        j += 1
        i = test_indices[j]

    test = index
    return i, true_index

def split_bayesian_change_point(frame_sizes, data, index, test_indices, threshold):
    frame_index, true_index = get_frame_indices(frame_sizes, index, test_indices)
    hashable_data = np.copy(data)
    hashable_data = hashable_data.tostring()
    if table.is_already_in_table(hashable_data):
        return frame_sizes, False
    frame_sizes = frame_sizes.copy()

    det = sdt.BayesOffline(obs_likelihood='ifm')
    features = []
    for _ in range(len(data[0])):
        features.append([])
    for i in range(frame_sizes[frame_index][true_index]):
        for j in range(len(data[i])):
            features[j].append(data[i][j])
    new_data = (np.transpose(features))[0]
    # new_data = np.empty(1)
    # for datum in data:
    #     if not new_data.any():
    #         new_data = datum
    #     else:
    #         new_data = np.concatenate((new_data, datum))
    #cut off 
    point_probability = det.find_changepoints(new_data)
    max_value = np.max(point_probability)
    index_of_change_point = np.argmax(point_probability)

    if max_value <= threshold: #Return if no change points are found
        table.save_value(hashable_data)
        return frame_sizes, False
    

    j = len(frame_sizes[frame_index])-1

    frame_sizes[frame_index].append(frame_sizes[frame_index][j])
    while(j > true_index):
        j -= 1
        frame_sizes[frame_index][j+1] = frame_sizes[frame_index][j]
    
    if frame_sizes[frame_index][j] - index_of_change_point <= 0:
        print('length of probs', len(point_probability))
        print('index of change point', index_of_change_point)
        print('old frame size: ', frame_sizes[frame_index][j])
        print('new frame size in j+1: ', frame_sizes[frame_index][j]-index_of_change_point)
    old_fs = frame_sizes[frame_index][j]
    frame_sizes[frame_index][j+1] = old_fs - index_of_change_point
    frame_sizes[frame_index][j] = index_of_change_point
  
    return frame_sizes, True
 

