# ###################################
# Group ID : 720
# Members : Aksel Porsborg Andersen, Magnus Stensli, Takekazu Gade Nakabayashi, Paula Lopez Urquia, Christian Lykke Joergensen
# Date : 8/10/2025
# Lecture: Lecture 6 Support vector machines
# Dependencies: numpy, matplotlib, sklearn, scipy, time, os
# Python version: 3.13.7
# Functionality: Perform classification for the entire MNIST dataset by using SVMs
# ###################################

# %%
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import os


# %% [markdown]
# # Exercise 7: Support Vector Machine (SVM) 
# Perform classification for the entire MNIST dataset by using SVMs, e.g. functions in Scikit-learn or Matlab.

# %% [markdown]
# ## Load data training and testing data

# %%
# Like exercise 6 (Copied the same code from previous exercise to load training/testing data)
# %%
def create_complete_datasets(folder_path):
    '''
    Function for creating complete training and test sets containing
    all classes
    '''
    # Empty lists
    trainset = []
    traintargets = []
    testset = []
    testtargets = []

    # For each class
    for i in range(10):
        train_file = os.path.join(folder_path, "train%d.txt" % i)
        test_file = os.path.join(folder_path, "test%d.txt" % i)
        train_data = np.loadtxt(train_file)
        test_data = np.loadtxt(test_file)
        trainset.append(train_data)
        traintargets.append(np.full(len(train_data), i))
        testset.append(test_data)
        testtargets.append(np.full(len(test_data), i))

    # Concatenate into to complete datasets
    train_set = np.concatenate(trainset, axis=0)
    traintargets = np.concatenate(traintargets)
    test_set = np.concatenate(testset, axis=0)
    testtargets = np.concatenate(testtargets)
    return train_set, traintargets, test_set, testtargets


# Change to your own path:
folder = r"C:\Users\Aksel\Desktop\Uni\Kandidat AVS\1 Semester\Machine Learning exercises\Lecture5ML\mnist_all"
# Complete training and test sets
train_set, train_targets, test_set, test_targets = create_complete_datasets(folder)

# Subsample training for 10%
num_train_samples = int(0.1 * len(train_set))
indices = np.random.choice(len(train_set), num_train_samples, replace=False)
train_set_sub = train_set[indices]
train_targets_sub = train_targets[indices] 
print(f"Subsample training set size: {train_set_sub.shape[0]} samples")

# %% [markdown]
# ## Create and fit SVM on training data

# Use sklearn
# (Please try training it:)
clf = svm.SVC(kernel='linear')
start_time = time.time()
clf.fit(train_set_sub, train_targets_sub)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} sec")
# %% [markdown]
# ## Test model on test set

# %%
# What is the accuracy on the test set?
startt_time = time.time()  # Using time.time() again to track how long it takes, since SVMs are slow..
predictions = clf.predict(test_set)
endd_time = time.time()
print(f"Prediction clf.predict time: {end_time - start_time:.2f} sec")

accuracy = np.mean(predictions == test_targets)
print(f"Test accuracy: {accuracy:.4f}")

# Write down Accuracy and training/clf.predict time:
# Accuracy: 0.9110
# Training time for 6000 samples: 0.91 sec

# %% [markdown]
# ## Plot Confusion matrix (use previous code from Lecture 5)
cm = confusion_matrix(test_targets, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix, SVM on MNIST (10% subsample)')
plt.show()


# %%
# Does the confusion matrix show us any insights about the model performance?
# Yes, it shows high diagonal values, indicating a good performance.

# %% [markdown]
# ## Comparing with PCA/LDA
# ## Plot Confusion matrix (use previous code from Lecture 5)
# The confusion matrix is attached "confusionMatrix-pcaLda.png"

# %%
# How does SVM compare to PCA and LDA (compare confusion matrices)
# The SVM confusion matrix shows a significantly better performance with a much higher accuracy compared to PCA and LDA.
# It is important to note that the SVM was only trained on 10%, where PCA and LDA were trained on the full MNIST dataset
