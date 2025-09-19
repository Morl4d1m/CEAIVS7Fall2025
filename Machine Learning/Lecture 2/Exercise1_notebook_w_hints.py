#!/usr/bin/env python
# coding: utf-8

# # Exercise
# ## You are given, as the train data, trn_x and trn_y along with their class labels trn_x_class and trn_y_class. The task is to classify the following TEST data.
# 

# First we load the data from the text files

# In[1]:


import numpy as np

# Train data
train_x = np.loadtxt("dataset1_G_noisy_ASCII/trn_x.txt")
train_x_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_x_class.txt")

train_y = np.loadtxt("dataset1_G_noisy_ASCII/trn_y.txt")
train_y_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_y_class.txt")

train_mean = np.mean(train_x, axis= 0)
train_var = np.var(train_x, axis= 0)

# Test data
test_x = np.loadtxt("dataset1_G_noisy_ASCII/tst_x.txt")
test_x_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_x_class.txt")

test_y = np.loadtxt("dataset1_G_noisy_ASCII/tst_y.txt")
test_y_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_y_class.txt")

test_y_126 = np.loadtxt("dataset1_G_noisy_ASCII/tst_y_126.txt")
test_y_126_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_y_126_class.txt")

test_xy = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy.txt")
test_xy_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_class.txt")

test_xy_126 = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126.txt")
test_xy_126_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126_class.txt")


# Looking at the data we see that our input features is 2-dimensional, i.e., it has two values per data point.
# Furthermore, x has label 1 and y has label 2.

# Let's visualize the training data by plotting a 2D scatter plot and corresponding Gaussians for class x and class y

# In[ ]:


# Hint: look at: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html 


# ### (a) classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;

# First we compute the statistics of x and y (use np.mean and np.cov)

# In[2]:


# x statistics
# train_x_mean = ?
# train_x_cov = ?

# y statistics
# train_y_mean = ?
# train_y_cov = ?

# priors
# prior_x = ?
# prior_y = ?


# Now we need to a function for computing the likelihood of x and y given our test data.

# In[3]:


# Define likelihood function
# Implement your own or look on stack overflow if you are lazy - most important thing is that you understand what is going on

# def likelihood(data, mean, cov):
#     likelihood_value = ? 
#     return likelihood_value


# To classify the test data we compute the likelihood of it being class x and class y

# In[4]:


# Compute likelihood of x and y


# We compute the posterior probability by taking the priors into account

# In[ ]:


# Compute posteriors from likelihood and prior


# Now choose to classify our test data as belonging to the class with the highest posterior probability

# In[5]:


# Remember that labels for x and y are are 1 and 2 respectively
# classification = ?


# We can compute the accuracy of our classifications by taking the sum of correct predictions and divide by the total number of predictions

# In[6]:


# accuracy_xy = ?


# ### (b) classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy;

# First we define our prior probabilities

# In[7]:


# prior_x_uniform = ?
# prior_y_uniform = ?


# We can now compute posteriors knowing that the posterior probability is simply the prior, p(C), multiplied by the likelihood p(x, C).

# In[8]:


# likelihood_x_uniform = ?
# likelihood_y_uniform = ?

# posterior_x_uniform = ?
# posterior_y_uniform = ?


# Now that we have posteriors for both x and y we can classify the test data and compute the accuracy

# In[3]:


# classification_uniform = ?

# accuracy_xy_126_uniform = ?
# print(f"Accuracy using uniform prior {accuracy_xy_126_uniform*100:.2f}%")


# ### (c) classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).

# Here we simply follow the procedure of (b), however, this time with updated priors

# In[1]:


# prior_x_non_uniform = ?
# prior_y_non_uniform = ?

# likelihood_x_non_uniform = ?
# likelihood_y_non_uniform = ?
# posterior_x_non_uniform = ?
# posterior_y_non_uniform = ?

# classification_non_uniform = ? 

# accuracy_xy_126_non_uniform = ?

# print(f"Accuracy using non-uniform prior {accuracy_xy_126_non_uniform*100:.2f}%")


# Comparing the accuracy using uniform prior and non-uniform priors we see that using prior information about the data distribution improves classifcation accuracy by ?%.

# In[2]:


# improvement = (accuracy_xy_126_non_uniform / accuracy_xy_126_uniform) - 1
# print(f"Absolute improvement in accuracy {improvement*100:.2f}%")


# In[ ]:




