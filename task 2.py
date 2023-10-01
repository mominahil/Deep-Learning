#!/usr/bin/env python
# coding: utf-8

# ## Task 2 (using Scikit Learn) 

# In[11]:


import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Sequential, layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[17]:


class MLPModel:
    
    # constructor for initializing the MLPModel class, with parameters num of features and total classes
    def __init__(self, shapeOfData, totalClasses):
        self.shapeOfData = shapeOfData
        self.totalClasses = totalClasses

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

    # making a KerasClassifier by calling model func and the specified hyperparameters
    def create_classifier(self, hidden_layers=2, neurons_per_layer=128, activation='relu'):
        return KerasClassifier(build_fn=self.build_model, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer, 
                               activation=activation,
                               epochs=20,  
                               batch_size=32 
                              )


# In[18]:


# Load the data
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


# In[19]:


# Initialize the MLPModel
input_shape = X_train.shape[1]
num_classes = len(np.unique(labels))
mlp_model = MLPModel(input_shape, num_classes)


# In[20]:


# Experiment 1 = no Hyperparameter Tuning
print("Experiment 1: Without Hyperparameter Tuning")
classifier = mlp_model.create_classifier()
history = classifier.fit(X_train, y_train, validation_data=(X_val, y_val))


# In[21]:


y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print("Confusion Matrix:")
print(cm)


# In[22]:


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[23]:


# Experiment 2 = Hyperparameter Tuning
print("\nExperiment 2: With Hyperparameter Tuning")
param_grid = {
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [32, 64], 
    'activation': ['relu', 'sigmoid'],
    'epochs': [20],
    'batch_size': [16, 32], 
}

classifier = mlp_model.create_classifier()
grid_search = GridSearchCV(classifier, param_grid, cv=2, verbose=1, n_jobs=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[26]:


# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print("Confusion Matrix:")
print(cm)


# In[27]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show() 


# In[ ]:




