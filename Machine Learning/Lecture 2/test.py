# ################################### 
# Group ID : 343
# Members : Johan Theil, Marcus Hodal, Mikkel Nielsen 
# Date : 2025/09/12 
# Lecture: 2 Bayesian decision theory
# Dependencies: Virtual enviroment
# Python version: 3
# Functionality: Short Description. This script shows overfitting, 
# ridge regression, and sample dependency 
# ################################### 
#!/usr/bin/env python
# coding: utf-8

# # Exercise
# ## You are given, as the train data, trn_x and trn_y along with their class labels trn_x_class and trn_y_class. 
# The task is to classify the following TEST data.
# 

# First we load the data from the text files

## In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
folder = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Machine Learning\Lecture 2\Fdataset1_G_noisy_ASCII"
os.chdir(folder)
# Train data
train_x = np.loadtxt("trn_x.txt")
train_x_label = np.loadtxt("trn_x_class.txt")

train_y = np.loadtxt("trn_y.txt")
train_y_label = np.loadtxt("trn_y_class.txt")

train_mean = np.mean(train_x, axis= 0)
train_var = np.var(train_x, axis= 0)

# Test data
test_x = np.loadtxt("tst_x.txt")
test_x_label = np.loadtxt("tst_x_class.txt")

test_y = np.loadtxt("tst_y.txt")
test_y_label = np.loadtxt("tst_y_class.txt")

test_y_126 = np.loadtxt("tst_y_126.txt")
test_y_126_label = np.loadtxt("tst_y_126_class.txt")

test_xy = np.loadtxt("tst_xy.txt")
test_xy_label = np.loadtxt("tst_xy_class.txt")

test_xy_126 = np.loadtxt("tst_xy_126.txt")
test_xy_126_label = np.loadtxt("tst_xy_126_class.txt")


# Looking at the data we see that our input features is 2-dimensional, i.e., it has two values per data point.
# Furthermore, x has label 1 and y has label 2.

# Let's visualize the training data by plotting a 2D scatter plot and corresponding Gaussians for class x and class y

## In[ ]:

# Hint: look at: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html 
colors = ['tab:blue', 'tab:orange']

fig, ax = plt.subplots()
ax.scatter(train_x[:,0], train_x[:,1], 
            c=colors[0], label='x',
            alpha=0.3, edgecolors='none')

ax.scatter(train_y[:,0], train_y[:,1], 
            c=colors[1], label='Y',
            alpha=0.3, edgecolors='none')

ax.legend()
ax.grid(True)

#plt.show()



# ### (a) classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;

# First we compute the statistics of x and y (use np.mean and np.cov)

## In[2]:


# x statistics
train_x_mean = np.mean(train_x, axis=0)
train_x_cov  = np.cov(train_x, rowvar=False) 

# y statistics
train_y_mean = np.mean(train_y, axis=0)
train_y_cov  = np.cov(train_y, rowvar=False)

# priors
prior_x = len(train_x_label)/(len(train_x_label)+len(train_y_label))
prior_y = len(train_y_label)/(len(train_x_label)+len(train_y_label))
# print(f"Prior_x = {prior_x}, prior_y = {prior_y}")


# Now we need to a function for computing the likelihood of x and y given our test data.

## In[3]:


# Define likelihood function
# Implement your own or look on stack overflow if you are lazy - most important thing is that you understand what is going on

def likelihood(data, mean, cov):
    d = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / (np.power((2 * np.pi), d/2) * np.sqrt(det_cov))

    diff = data - mean
    # Mahalanobis distance
    exponent = -0.5 * np.einsum('...i,ij,...j', diff, inv_cov, diff)
    return norm_const * np.exp(exponent)


# To classify the test data we compute the likelihood of it being class x and class y

## In[4]:


likelihood_x = likelihood(test_xy, train_x_mean, train_x_cov)
likelihood_y = likelihood(test_xy, train_y_mean, train_y_cov)
print(f"likelihoodx = {likelihood_x}, likelihoody = {likelihood_y}")


# We compute the posterior probability by taking the priors into account

## In[ ]:


posterior_x = prior_x * likelihood_x
posterior_y = prior_y * likelihood_y
print(f"likelihood post x = {posterior_x}, likelihood post y = {posterior_y}")

# Now choose to classify our test data as belonging to the class with the highest posterior probability

## In[5]:


# Remember that labels for x and y are are 1 and 2 respectively
classification = np.where(posterior_x > posterior_y, 1, 2)
accuracy_xy = np.mean(classification == test_xy_label)
print(f"(a) Accuracy on tst_xy: {accuracy_xy*100:.2f}%")


# We can compute the accuracy of our classifications by taking the sum of correct predictions and divide by the total number of predictions

## In[6]:

## In[7]:


prior_x_uniform = len(train_x_label)/(len(train_x_label)+len(train_y_label))
prior_y_uniform = len(train_y_label)/(len(train_x_label)+len(train_y_label))

likelihood_x_uniform = likelihood(test_xy_126, train_x_mean, train_x_cov)
likelihood_y_uniform = likelihood(test_xy_126, train_y_mean, train_y_cov)

posterior_x_uniform = prior_x_uniform * likelihood_x_uniform
posterior_y_uniform = prior_y_uniform * likelihood_y_uniform

classification_uniform = np.where(posterior_x_uniform > posterior_y_uniform, 1, 2)
accuracy_xy_126_uniform = np.mean(classification_uniform == test_xy_126_label)
print(f"(b) Accuracy with uniform prior: {accuracy_xy_126_uniform*100:.2f}%")

# We can now compute posteriors knowing that the posterior probability is simply the prior, p(C), multiplied by the likelihood p(x, C).

## In[8]:


prior_x_non_uniform = 0.9
prior_y_non_uniform = 0.1

likelihood_x_non_uniform = likelihood(test_xy_126, train_x_mean, train_x_cov)
likelihood_y_non_uniform = likelihood(test_xy_126, train_y_mean, train_y_cov)

posterior_x_non_uniform = prior_x_non_uniform * likelihood_x_non_uniform
posterior_y_non_uniform = prior_y_non_uniform * likelihood_y_non_uniform

classification_non_uniform = np.where(posterior_x_non_uniform > posterior_y_non_uniform, 1, 2)
accuracy_xy_126_non_uniform = np.mean(classification_non_uniform == test_xy_126_label)
print(f"(c) Accuracy with non-uniform prior (0.9/0.1): {accuracy_xy_126_non_uniform*100:.2f}%")


# Comparing the accuracy using uniform prior and non-uniform priors we see that using prior information about the data distribution improves classifcation accuracy by ?%.

## In[2]:

improvement = (accuracy_xy_126_non_uniform / accuracy_xy_126_uniform) - 1
print(f"Improvement over uniform prior: {improvement*100:.2f}%")


## In[ ]: