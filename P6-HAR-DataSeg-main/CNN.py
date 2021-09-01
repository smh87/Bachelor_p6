import tensorflow as tf
from tensorflow.core.protobuf.config_pb2 import RunOptions
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import numpy as np
import os
import matplotlib.pyplot as plt
import gc
from dotenv import load_dotenv

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

load_dotenv()
class Cnn:
  def __init__(self, activities, X_train):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    self.model = Sequential()

    self.model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
    self.model.add(Dropout(0.1))

    self.model.add(Conv2D(32, (2, 2), activation='relu'))
    self.model.add(Dropout(0.2))

    self.model.add(Flatten())

    self.model.add(Dense(64, activation = 'relu'))
    self.model.add(Dropout(0.5))

    self.model.add(Dense(activities, activation='softmax'))
    RunOptions(report_tensor_allocations_upon_oom = True)
    self.model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

  def plot_learningCurve(self, history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.close('all')

  def run_cnn(self, X_train, X_test, y_train, y_test, label, depth):
    print(set(y_test))
    print('test labels are ', label.inverse_transform(list(set(y_test))))
    epochs = int(os.getenv('EPOCHS'))
    history = self.model.fit(X_train, y_train, epochs = epochs, validation_data= (X_test, y_test), verbose=1)

    #self.plot_learningCurve(history, 10)

    y_pred = self.model.predict(X_test)
    all_data = np.concatenate((X_train, X_test))
    collected_pred = self.model.predict(all_data)
    results = np.argmax(collected_pred, axis=1)
    
    # mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
    
    # plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))
    # plt.savefig('confusion_matrix'+str(depth)+'.png')
    #plt.show()
    plt.close('all')
    gc.collect()
    return results, self.getConfidence(collected_pred, results) 

  def getConfidence(self, y_pred, results):
    confidence = np.zeros(len(results))
    for i in range(len(y_pred)):
      confidence[i] = y_pred[i][results[i]]
    return confidence

 