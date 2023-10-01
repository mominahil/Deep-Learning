#!/usr/bin/env python
# coding: utf-8

# ## Task 1

# In[52]:


import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Sequential, layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[53]:


class MLPModel:
    
    # constructor for initializing the MLPModel class, with parameters num of features and total classes
    def __init__(self, shapeOfData, totalClasses):
        self.shapeOfData = shapeOfData
        self.totalClasses = totalClasses
        self.model = self.build_model()

        
    # function for building the MLP model, with layers, neurons and activation func as parameters
    def build_model(self, hidden_layers=2, neurons_per_layer=128, activation='relu'):
        
        # making a sequential model (linear stack of layers)
        model = Sequential()
        # adding input layer with shapeOfData 
        model.add(layers.Input(shape=self.shapeOfData))
        
        # addin hidden layers to model
        # dense means fully connected 
        for _ in range(hidden_layers):
            model.add(layers.Dense(neurons_per_layer, activation=activation))
        
        # using softmax func for multiclass classification
        model.add(layers.Dense(self.totalClasses, activation='softmax'))
        
        # compilimg model for trainng 
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # func to train the model
    def fit(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        
        # training model on training data and validating on validation data 
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        return history

    # func to test the trained model
    def test(self, X_test, y_test):
        
        # prediting class probabilites for test data 
        y_pred = self.model.predict(X_test)
        
        # selecting the class with the highest probability 
        # connverting class probabalities to class labels
        y_pred_classes = np.argmax(y_pred, axis=1)

        # calculating results
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='macro')
        recall = recall_score(y_test, y_pred_classes, average='macro')
        f1 = f1_score(y_test, y_pred_classes, average='macro')
        cm = confusion_matrix(y_test, y_pred_classes)

        return accuracy, precision, recall, f1, cm
    
    
    def create_classifier(self):
        return KerasClassifier(build_fn=self.build_model)

    def tune(self, X_train, y_train, X_val, y_val, param_grid, cv, n_jobs):
        classifier = self.create_classifier()
        # tunimg hyperparameters by gridsearchcv
        grid_search = GridSearchCV(classifier, param_grid, cv=cv, verbose=1, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        # getting the best hyperparameters
        best_params = grid_search.best_params_
        # getting best estimator
        best_model = grid_search.best_estimator_
        # getting validation score
        val_score = best_model.score(X_val, y_val)
        
        return best_params, val_score


# In[54]:


data = []  # List to store image data
labels = []  # List to store labels

# Define a function to extract the person label from the file name
def extract_person_label(file_name):
    return int(file_name.split('.')[0].replace('subject', '')) - 1  # Subtract 1 to make labels start from 0

# Load data
dataset_dir = 'C:/Users/Hp/Jupyter/yale face/data'

for file_name in os.listdir(dataset_dir):
    img = plt.imread(os.path.join(dataset_dir, file_name))
    data.append(img.flatten())  # Flatten image into a 1D array
    labels.append(extract_person_label(file_name))

data = np.array(data)
labels = np.array(labels)

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[55]:


input_shape = X_train.shape[1]
num_classes = len(np.unique(labels))
mlp_model = MLPModel(input_shape, num_classes)


# In[56]:


print("Experiment 1= no Hyperparameter Tuning")
history = mlp_model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)


# In[57]:


accuracy, precision, recall, f1, cm = mlp_model.test(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print("Confusion Matrix:")
print(cm)


# In[58]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[59]:


print("\nExperiment 2: With Hyperparameter Tuning")
param_grid = {
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [32, 64], 
    'activation': ['relu', 'sigmoid'],
    'epochs': [20],
    'batch_size': [16, 32], 
}

best_params, val_score = mlp_model.tune(X_train, y_train, X_val, y_val, param_grid, cv=2, n_jobs=1)

print("Best Hyperparameters:", best_params)
print(f'Validation Score with Best Model: {val_score:.2f}')


# In[60]:


best_model = MLPModel(input_shape, num_classes)
best_model.model = best_model.build_model(hidden_layers=best_params['hidden_layers'],
                                          neurons_per_layer=best_params['neurons_per_layer'],
                                          activation=best_params['activation'])
history = best_model.fit(X_train, y_train, X_val, y_val, epochs=best_params['epochs'], batch_size=best_params['batch_size'])


# In[61]:


accuracy, precision, recall, f1, cm = best_model.test(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print("Confusion Matrix:")
print(cm)


# In[62]:


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:




