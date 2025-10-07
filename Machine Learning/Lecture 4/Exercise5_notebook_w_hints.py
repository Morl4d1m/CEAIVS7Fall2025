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
data_path = "2D568class.mat"
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

# %%
# We know that our data contains 3 clusters:
clusters=3
# And by knowing we have 3 distinct clusters beforehand, we can also specify the covariance type to be "Full"
covType='full'
# And by reading the documentation, random_state should be set to some int, to ensure repoducible results:
# Therefore we call:
gmm = GMM(n_components=clusters, covariance_type=covType, random_state=0)
# And we would like to fit our training data to it:
gmm.fit(trainset)
# Now, the means and covariances can be extrapolated:
gmmMeans = gmm.means_ #Oops, this is too early - nevertheless, its kept here as well
print('Mean of each cluster in GMM: \n', gmmMeans, '\n')
gmmCovs = gmm.covariances_
print('Covariances of each cluster in GMM: \n', gmmCovs, '\n')

# %% [markdown]
# ## Creating Gaussian models
# Following the same approach from the previous exercises we can also estimate Gaussian models for each class

# %% 
# Assign means to each class - this will make it more overseeable than the previous prints, and show the actual class means, and not the GMM means
mean5 = np.mean(train5, axis=0)
mean6 = np.mean(train6, axis=0)
mean8 = np.mean(train8, axis=0)
print("Class means:")
print("5:", mean5)
print("6:", mean6)
print("8:", mean8, '\n')
# Assign covariances to each class
cov5 = np.cov(train5, rowvar=False)
cov6 = np.cov(train6, rowvar=False)
cov8 = np.cov(train8, rowvar=False)
print("Class covariances:")
print("5:\n", cov5)
print("6:\n", cov6)
print("8:\n", cov8, '\n')



# %% [markdown]
# ## Comparing means and covariance matrices.
# Let's look at the means and covariance matrices.
#
# First we extract the means and covariances from the GMM.
# Basically dividing gmmMeans and gmmCovs into separate entities:
meanGmm1, meanGmm2, meanGmm3 = gmmMeans # Python notation is weird when you're used to embedded C
covGmm1, covGmm2, covGmm3 = gmmCovs

# Means comparison:
for name, mean in {
    "Class 5": mean5, "Class 6": mean6, "Class 8": mean8,
    "GMM 1": meanGmm1, "GMM 2": meanGmm2, "GMM 3": meanGmm3}.items():
    print(f"{name}: {np.array2string(mean, precision=3)}")

# Covariance comparison:
for name, cov in {
    "Class 5": cov5, "Class 6": cov6, "Class 8": cov8,
    "GMM 1": covGmm1, "GMM 2": covGmm2, "GMM 3": covGmm3}.items():
    print(f"{name}:\n{cov}\n")

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
"""
# Class covariances
axs[0, 0].matshow(cov5, cmap="viridis")
axs[0, 0].set_title("Cov. Class 5")
axs[0, 1].matshow(cov6, cmap="viridis")
axs[0, 1].set_title("Cov. Class 6")
axs[0, 2].matshow(cov8, cmap="viridis")
axs[0, 2].set_title("Cov. Class 8")

# GMM covariances
axs[1, 0].matshow(covGmm1, cmap="viridis")
axs[1, 0].set_title("Cov. GMM 1")
axs[1, 1].matshow(covGmm2, cmap="viridis")
axs[1, 1].set_title("Cov. GMM 2")
axs[1, 2].matshow(covGmm3, cmap="viridis")
axs[1, 2].set_title("Cov. GMM 3")

plt.tight_layout()
plt.show()

# Create grid
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

plt.figure(figsize=(12, 10))
plt.scatter(trainset[:, 0], trainset[:, 1], s=10, alpha=0.3, label="Data")

# Class Gaussians
for mean, cov, color, label in zip(
    [mean5, mean6, mean8],
    [cov5, cov6, cov8],
    ["blue", "green", "orange"],
    ["Class 5", "Class 6", "Class 8"]
):
    rv = norm(mean, cov)
    plt.contour(X, Y, rv.pdf(pos), levels=5, colors=color)
    plt.scatter(mean[0], mean[1], c=color, marker="o", s=100, label=label)

# GMM components
for i, (mean, cov) in enumerate(zip(gmmMeans, gmmCovs)):
    rv = norm(mean, cov)
    plt.contour(X, Y, rv.pdf(pos), levels=5, linestyles="dashed", colors="red")
    plt.scatter(mean[0], mean[1], c="red", marker="x", s=100, label=f"GMM {i+1}")

plt.title("GMM vs Class Gaussians (Contours)")
plt.legend()
plt.show()
"""

# %% [markdown]
# Now we can compare the GMM means and covariances to the Gaussin models estimated for each class individually.

# %% [markdown]
# ### Means

# %%
# for name, mean in {"mean5": mean5, "mean6": mean6, "mean8": mean8,
#                    "mean1_gmm": mean1_gmm, "mean2_gmm": mean2_gmm, "mean3_gmm": mean3_gmm}.items():
#     print(f"{name}: {np.array2string(mean)}")

# %% [markdown]
# ### Covariances

# %%
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].matshow(cov5)
for (i, j), z in np.ndenumerate(cov5):
    axs[0, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 0].set_title("Cov. Class: 5")

axs[1, 0].matshow(covGmm1)
for (i, j), z in np.ndenumerate(covGmm1):
    axs[1, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 0].set_title("Cov. GMM kernel 1")

axs[0, 1].matshow(cov6)
for (i, j), z in np.ndenumerate(cov6):
    axs[0, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 1].set_title("Cov. Class: 6")

axs[1, 1].matshow(covGmm2)
for (i, j), z in np.ndenumerate(covGmm2):
    axs[1, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 1].set_title("Cov. GMM kernel 2")

axs[0, 2].matshow(cov8)
for (i, j), z in np.ndenumerate(cov8):
    axs[0, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 2].set_title("Cov. Class: 8")

c = axs[1, 2].matshow(covGmm1)
for (i, j), z in np.ndenumerate(covGmm3):
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
# Create points to do a contour a plot

# %% [markdown]
# We can also visualize the separate Gaussian models from the GMM by creating Gassians from the classwise means and covariances of the GMM model.

# %%


# %% [markdown]
# Now we sample from the models using the generated points.

# %%


# %% [markdown]
# The model samples can then be visualized in a contour plot.

# %%
# Plot contours for the GMM, seperated GMM and individual estimated densities

# %%
