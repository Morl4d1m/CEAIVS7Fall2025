# ###################################
# Group ID : 720
# Members : Aksel Porsborg Andersen, Magnus Stensli, Takekazu Gade Nakabayashi, Paula Lopez Urquia, Christian Lykke Joergensen, Nusrat Nabi
# Date : 24/09/2025
# Lecture: Lecture 4 Clustering
# Dependencies: numpy, matplotlib, sklearn, scipy
# Python version: 3.13.7
# Functionality: Comparing Gaussian Mixture Models with Gaussian models
# ###################################
# %%
from scipy.io import loadmat
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

# %% [markdown]
# # Exercise 5: Clustering
# This assignment is based on the previously generated 2-dimensional data of the three classes (5, 6 and 8) from the MNIST database of handwritten digits.
#
# First, mix the 2-dimensional data (training data only) by removing the labels and then use one Gaussian mixture model to model them.
#
# Secondly, compare the Gaussian mixture model with the Gaussian models trained in the previous assignment, in terms of mean and variance values as well as through visualisation.

# %% [markdown]
# ## Loading the data and mixing
# First we load the exercise data set, combine the individual training sets into one and shuffle the data to ensure a random shuffle (here with a seed to ensure reproducability).

# %%
data_path = "C:\\Users\\plope\\Documents\\AAUFall\\ML\\Topic 5 (Lecture 4)\\2D568class.mat" # Change this to your path
data = loadmat(data_path)
train5 = data["trn5_2dim"] / 255
train6 = data["trn6_2dim"] / 255
train8 = data["trn8_2dim"] / 255

trainset = np.concatenate([train5, train6, train8])
np.random.seed(0)
np.random.shuffle(trainset)

# %% [markdown]
# ## Creating a Gaussian Mixture model
# First create a Gaussian Mixture Model of the data using sklearn

gmm = GMM(n_components=3, covariance_type='full', random_state=0)
gmm.fit(trainset)
labels = gmm.predict(trainset)
probs = gmm.predict_proba(trainset)
print("Means:\n", gmm.means_)
print("Covariances:\n", gmm.covariances_)

# %% 
# Plotting the results
# plt.figure(figsize=(12, 4))

# # Plot 1: Original data points
# plt.subplot(1, 3, 1)
# plt.scatter(trainset[:, 0], trainset[:, 1], alpha=0.6, s=30)
# plt.title('Original Mixed Data')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# # Plot 2: Data colored by GMM cluster assignments
# plt.subplot(1, 3, 2)
# colors = ['red', 'blue', 'green']
# for i in range(3):
#     cluster_points = trainset[labels == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                c=colors[i], alpha=0.6, s=30, label=f'Cluster {i}')
# plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
#            c='black', marker='x', s=200, linewidth=3, label='Centroids')
# plt.title('GMM Clustering Results')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()

# # Plot 3: Probability contours
# plt.subplot(1, 3, 3)
# x_min, x_max = trainset[:, 0].min() - 0.1, trainset[:, 0].max() + 0.1
# y_min, y_max = trainset[:, 1].min() - 0.1, trainset[:, 1].max() + 0.1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                      np.linspace(y_min, y_max, 100))
# grid_points = np.c_[xx.ravel(), yy.ravel()]
# Z = gmm.score_samples(grid_points)
# Z = Z.reshape(xx.shape)

# plt.contour(xx, yy, Z, levels=10)
# plt.scatter(trainset[:, 0], trainset[:, 1], c=labels, 
#            cmap='viridis', alpha=0.6, s=30)
# plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
#            c='red', marker='x', s=200, linewidth=3)
# plt.title('GMM Density Contours')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# plt.tight_layout()
# plt.show()

# # Print some statistics
# print(f"\nDataset shape: {trainset.shape}")
# print(f"Number of points per cluster:")
# for i in range(3):
#     count = np.sum(labels == i)
#     print(f"  Cluster {i}: {count} points ({count/len(trainset)*100:.1f}%)")

# print(f"\nCluster means:")
# for i, mean in enumerate(gmm.means_):
#     print(f"  Cluster {i}: [{mean[0]:.3f}, {mean[1]:.3f}]")


# %% [markdown]
# ## Creating Gaussian models
# Following the same approach from the previous exercises we can also estimate Gaussian models for each class

# %% [markdown]
# ## Comparing means and covariance matrices.
# Let's look at the means and covariance matrices.
#
# First we extract the means and covariances from the GMM.
mean5 = np.mean(train5, axis=0)
cov5 = np.cov(train5, rowvar=False)

mean6 = np.mean(train6, axis=0)
cov6 = np.cov(train6, rowvar=False)

mean8 = np.mean(train8, axis=0)
cov8 = np.cov(train8, rowvar=False)


# %% [markdown]
# Now we can compare the GMM means and covariances to the Gaussin models estimated for each class individually.

mean1_gmm = gmm.means_[0]
cov1_gmm = gmm.covariances_[0]
mean2_gmm = gmm.means_[1]
cov2_gmm = gmm.covariances_[1]
mean3_gmm = gmm.means_[2]
cov3_gmm = gmm.covariances_[2]

# %% [markdown]
# ### Means

# %%
for name, mean in {"mean5": mean5, "mean6": mean6, "mean8": mean8,
                   "mean1_gmm": mean1_gmm, "mean2_gmm": mean2_gmm, "mean3_gmm": mean3_gmm}.items():
    print(f"{name}: {np.array2string(mean)}")

# %% [markdown]
# ### Covariances

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].matshow(cov5)
for (i, j), z in np.ndenumerate(cov5):
    axs[0, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 0].set_title("Cov. Class: 5")

axs[1, 0].matshow(cov1_gmm)
for (i, j), z in np.ndenumerate(cov1_gmm):
    axs[1, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 0].set_title("Cov. GMM kernel 1")

axs[0, 1].matshow(cov6)
for (i, j), z in np.ndenumerate(cov6):
    axs[0, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 1].set_title("Cov. Class: 6")

axs[1, 1].matshow(cov2_gmm)
for (i, j), z in np.ndenumerate(cov2_gmm):
    axs[1, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 1].set_title("Cov. GMM kernel 2")

axs[0, 2].matshow(cov8)
for (i, j), z in np.ndenumerate(cov8):
    axs[0, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 2].set_title("Cov. Class: 8")

c = axs[1, 2].matshow(cov3_gmm)
for (i, j), z in np.ndenumerate(cov3_gmm):
    axs[1, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 2].set_title("Cov. GMM kernel 3")

plt.tight_layout()
plt.show()

# %% [markdown]
# What do we see when comparing means and covariances?

# %% [markdown]
# ## Visualizing the models in contourplots.
# Now we would like to visualize our models to compare them.

# %% [markdown]
# We first generate some points to be able to sample from the models.

# %%
# Create points to do a contour plot
x_min, x_max = trainset[:, 0].min() - 1, trainset[:, 0].max() + 1
y_min, y_max = trainset[:, 1].min() - 1, trainset[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# %% [markdown]
# We can also visualize the separate Gaussian models from the GMM by creating Gaussians from the classwise means and covariances of the GMM model.

# %%
# Create individual Gaussian models from supervised learning
gauss5 = norm(mean5, cov5)
gauss6 = norm(mean6, cov6)
gauss8 = norm(mean8, cov8)

# Create individual Gaussian models from GMM components
gauss1_gmm = norm(mean1_gmm, cov1_gmm)
gauss2_gmm = norm(mean2_gmm, cov2_gmm)
gauss3_gmm = norm(mean3_gmm, cov3_gmm)

# %% [markdown]
# Now we sample from the models using the generated points.

# %%
# Sample from supervised Gaussian models
Z_gauss5 = gauss5.pdf(grid_points).reshape(xx.shape)
Z_gauss6 = gauss6.pdf(grid_points).reshape(xx.shape)
Z_gauss8 = gauss8.pdf(grid_points).reshape(xx.shape)

# Sample from GMM individual components
Z_gmm1 = gauss1_gmm.pdf(grid_points).reshape(xx.shape)
Z_gmm2 = gauss2_gmm.pdf(grid_points).reshape(xx.shape)
Z_gmm3 = gauss3_gmm.pdf(grid_points).reshape(xx.shape)

# Sample from full GMM (mixture)
Z_gmm_full = np.exp(gmm.score_samples(grid_points)).reshape(xx.shape)

# %% [markdown]
# The model samples can then be visualized in a contour plot.

# %%
# Plot contours for the GMM, separated GMM and individual estimated densities
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Top row: Supervised Gaussian models
axes[0, 0].contour(xx, yy, Z_gauss5, levels=8, colors='blue')
axes[0, 0].scatter(train5[:, 0], train5[:, 1], alpha=0.3, s=10, c='blue')
axes[0, 0].scatter(mean5[0], mean5[1], c='red', marker='x', s=200, linewidth=3)
axes[0, 0].set_title('Supervised: Class 5')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

axes[0, 1].contour(xx, yy, Z_gauss6, levels=8, colors='green')
axes[0, 1].scatter(train6[:, 0], train6[:, 1], alpha=0.3, s=10, c='green')
axes[0, 1].scatter(mean6[0], mean6[1], c='red', marker='x', s=200, linewidth=3)
axes[0, 1].set_title('Supervised: Class 6')
axes[0, 1].set_xlabel('Feature 1')

axes[0, 2].contour(xx, yy, Z_gauss8, levels=8, colors='orange')
axes[0, 2].scatter(train8[:, 0], train8[:, 1], alpha=0.3, s=10, c='orange')
axes[0, 2].scatter(mean8[0], mean8[1], c='red', marker='x', s=200, linewidth=3)
axes[0, 2].set_title('Supervised: Class 8')
axes[0, 2].set_xlabel('Feature 1')

# Combined supervised
axes[0, 3].contour(xx, yy, Z_gauss5, levels=6, colors='blue', alpha=0.7)
axes[0, 3].contour(xx, yy, Z_gauss6, levels=6, colors='green', alpha=0.7)
axes[0, 3].contour(xx, yy, Z_gauss8, levels=6, colors='orange', alpha=0.7)
axes[0, 3].scatter(trainset[:, 0], trainset[:, 1], alpha=0.3, s=10, c='gray')
axes[0, 3].set_title('All Supervised Models')
axes[0, 3].set_xlabel('Feature 1')

# Bottom row: GMM models
axes[1, 0].contour(xx, yy, Z_gmm1, levels=8, colors='purple')
axes[1, 0].scatter(trainset[labels == 0][:, 0], trainset[labels == 0][:, 1], alpha=0.3, s=10, c='purple')
axes[1, 0].scatter(mean1_gmm[0], mean1_gmm[1], c='red', marker='x', s=200, linewidth=3)
axes[1, 0].set_title('GMM: Component 1')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')

axes[1, 1].contour(xx, yy, Z_gmm2, levels=8, colors='cyan')
axes[1, 1].scatter(trainset[labels == 1][:, 0], trainset[labels == 1][:, 1], alpha=0.3, s=10, c='cyan')
axes[1, 1].scatter(mean2_gmm[0], mean2_gmm[1], c='red', marker='x', s=200, linewidth=3)
axes[1, 1].set_title('GMM: Component 2')
axes[1, 1].set_xlabel('Feature 1')

axes[1, 2].contour(xx, yy, Z_gmm3, levels=8, colors='magenta')
axes[1, 2].scatter(trainset[labels == 2][:, 0], trainset[labels == 2][:, 1], alpha=0.3, s=10, c='magenta')
axes[1, 2].scatter(mean3_gmm[0], mean3_gmm[1], c='red', marker='x', s=200, linewidth=3)
axes[1, 2].set_title('GMM: Component 3')
axes[1, 2].set_xlabel('Feature 1')

# Full GMM mixture
axes[1, 3].contour(xx, yy, Z_gmm_full, levels=10, colors='black')
axes[1, 3].scatter(trainset[:, 0], trainset[:, 1], c=labels, alpha=0.5, s=10, cmap='viridis')
axes[1, 3].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=200, linewidth=3)
axes[1, 3].set_title('Full GMM Mixture')
axes[1, 3].set_xlabel('Feature 1')

plt.tight_layout()
plt.show()

