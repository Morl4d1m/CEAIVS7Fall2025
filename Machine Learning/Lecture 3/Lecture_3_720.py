# ###################################
# Group ID : 720
# Members : Aksel Porsborg Andersen, Magnus Stensli, Takekazu Gade Nakabayashi, Paula Lopez Urquia, Christian Lykke Joergensen, Nusrat Nabi
# Date : 19/09/2025
# Lecture: Lecture 3 Dimensionality reduction, scipy
# Dependencies: numpy, matplotlib, sklearn
# Python version: 3.13.7
# Functionality: 
# ###################################

# %%
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# %% [markdown]
# # Loading the training and test data

# %% [markdown]
# ## Train data

# %% Change to correct path:
train5 = np.loadtxt("Lecture3 /mnist_all/train5.txt") / 255  # /255 for normalization
train6 = np.loadtxt("Lecture3 /mnist_all/train6.txt") / 255
train8 = np.loadtxt("Lecture3 /mnist_all/train8.txt") / 255

# %%
# Define targets
train5_target = 5 * np.ones(len(train5))
train6_target = 6 * np.ones(len(train6))
train8_target = 8 * np.ones(len(train8))

# %%
# Combine data
train_data = np.concatenate([train5, train6, train8])
train_targets = np.concatenate([train5_target, train6_target, train8_target])

# %% [markdown]
# ## Test data

# %% Change to correct path:
test5 = np.loadtxt("Lecture 3/mnist_all/test5.txt") / 255
test6 = np.loadtxt("Lecture 3/mnist_all/test6.txt") / 255
test8 = np.loadtxt("Lecture 3/mnist_all/test8.txt") / 255

# Define targets
test5_target = 5 * np.ones(len(test5))
test6_target = 6 * np.ones(len(test6))
test8_target = 8 * np.ones(len(test8))

# Combine
test_data = np.concatenate([test5, test6, test8])
test_targets = np.concatenate([test5_target, test6_target, test8_target])

# Class names
classes = np.array([5, 6, 8])

# %% [markdown]
# # Part 1: Reduce dimension to 2
# Here, we wish to reduce the data dimensionality from 784 to 2 using Linear Disicriminant Analysis (LDA).
# For this you can use scikit-learn. The LDA class in scikit-learn fits a covariance matrix and compute eigenvectors for you. LDA assume that you know about the classes, so you have to use the concatenated training set and targets/classes.

# %%
# Fit a scikit learn LDA instance to training data
LDA_model = LDA(n_components=2)
LDA_model.fit(train_data, train_targets)

# %%
# Transform train data from each class using fitted LDA instance
train_data_LDA5 = LDA_model.transform(train5)
train_data_LDA6 = LDA_model.transform(train6)
train_data_LDA8 = LDA_model.transform(train8)

# %% [markdown]
# # Part 2: Perform 3-class classification based on the generated 2-dimensional data.
# We need to find a model to classify the test data as either 5, 6, or 8.
# Here, we could use a Gaussian model for each class, and estimate the mean and covariance from the dimensionality reduced data.

# %% [markdown]
# ## Estimate Gaussians using 2-dimensional data obtained from LDA
gaussian_5 = norm(mean=np.mean(train_data_LDA5, axis=0), cov=np.cov(train_data_LDA5, rowvar=False))
gaussian_6 = norm(mean=np.mean(train_data_LDA6, axis=0), cov=np.cov(train_data_LDA6, rowvar=False))
gaussian_8 = norm(mean=np.mean(train_data_LDA8, axis=0), cov=np.cov(train_data_LDA8, rowvar=False))

# %%
# Estimate parameters for a bivariante Gaussian distribution.
mean_5 = np.mean(train_data_LDA5, axis=0)
cov_5 = np.cov(train_data_LDA5, rowvar=False)
mean_6 = np.mean(train_data_LDA6, axis=0)
cov_6 = np.cov(train_data_LDA6, rowvar=False)
mean_8 = np.mean(train_data_LDA8, axis=0)
cov_8 = np.cov(train_data_LDA8, rowvar=False)

# %% [markdown]
# ## Classifying test data
# To classify the test data, we first transform it to 2-dimensions as well.

# %%
# Transform test data using fitted LDA instance
test_data_LDA = LDA_model.transform(test_data)

# %% [markdown]
# Now we compute priors, likelihoods and posteriors

# %%
# Compute priors
prior_5 = len(train5) / len(train_data)
prior_6 = len(train6) / len(train_data)
prior_8 = len(train8) / len(train_data)

# Compute Likelihoods
likelihood_5 = gaussian_5.pdf(test_data_LDA)
likelihood_6 = gaussian_6.pdf(test_data_LDA)
likelihood_8 = gaussian_8.pdf(test_data_LDA)

# Compute posteriors
posterior_5 = likelihood_5 * prior_5
posterior_6 = likelihood_6 * prior_6
posterior_8 = likelihood_8 * prior_8

# %% [markdown]
# We can now compute the classification accuracy for the LDA-dimensionality reduced data

# %%
# Compute predictions
predictions = np.argmax(np.vstack([posterior_5, posterior_6, posterior_8]), axis=0)
predicted_classes = classes[predictions]

# Compute accuracy
accuracy = np.mean(predicted_classes == test_targets)
print(f"Classification accuracy using LDA: {accuracy * 100:.2f}%")

# %% [markdown]
# What does the results show?
# 98.48% for its classification accuracy using LDA. 


# %% [markdown]
# # (Optional Task) Comparison with PCA

# %% [markdown]
# This (optional!) task involves reducing the dimensionality of the data instead using PCA in order to compare it with LDA.

# %%
from sklearn.decomposition import PCA

# %% [markdown]
# ## Part 1

# %% [markdown]
# Optionally also fit The PCA class in scikit-learn fits a covariance matrix and compute eigenvectors for you.
# PCA doesn't assume any knowledge about the classes, so you have to use the concatenated training set.

# %%
# Fit a scikit learn PCA instance to training data

# %% [markdown]
# Now that the PCA model is fit to the training data, we can find a low dimesional representation of each class.

# %% [markdown]
# Let's try to plot the dimensionality reduced data and compare PCA to LDA. What do we see?

# %%
# Scatter plot of the dimensional-reduced data

# %%
# Transform train data from each class using fitted PCA instance

# %% [markdown]
# In the above plot we should see that LDA is seemingly better at seperating the tree classes,while the classes 5 and 8 are highly overlapped when using PCA.

# %% [markdown]
# ## Estimate Gaussians using 2-dimensional data obtained from PCA

# %%
# Estimate parameters for a bivariante Gaussian distribution.

# %% [markdown]
# ## Classifying test data
# To classify the test data, we first transform it to 2-dimensions as well.

# %%
# Transform test data using fitted PCA/LDA instance

# %% [markdown]
# Now we compute priors, likelihoods and posteriors

# %%
# Compute priors
# Compute Likelihoods
# Compute posteriors

# %%
# Compute predictions

# Compute accuracy

# %% [markdown]
# We can now compare the classification accuracy from PCA and LDA. What does the results show?
