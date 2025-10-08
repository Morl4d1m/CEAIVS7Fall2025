# ###################################
# Group ID : 720
# Members : Aksel Porsborg Andersen, Magnus Stensli, Takekazu Gade Nakabayashi, Paula Lopez Urquia, Christian Lykke Joergensen
# Date : 07/10/2025
# Lecture: Lecture 5 Linear discrimination
# Dependencies: numpy, matplotlib, sklearn, scipy
# Python version: 3.13.7
# Functionality: Perform classification for the entire MNIST dataset
# ###################################

# %%
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


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

# %% default value: 9 components
n_components = 2

# PCA
pca = PCA(n_components=n_components)
train_pca = pca.fit_transform(train_set)
test_pca = pca.transform(test_set)

# LDA
lda = LDA(n_components=n_components)
train_lda = lda.fit_transform(train_set, train_targets)
test_lda = lda.transform(test_set)


# %%
# Analyze proportion of Variance.
print("PCA cumulative explained variance ratio:", np.sum(pca.explained_variance_ratio_))

if n_components == 2:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    scatter_pca = axs[0].scatter(train_pca[:, 0], train_pca[:, 1], c=train_targets, cmap='tab10')
    axs[0].set_title('PCA 2D Visualization')
    plt.colorbar(scatter_pca, ax=axs[0])

    scatter_lda = axs[1].scatter(train_lda[:, 0], train_lda[:, 1], c=train_targets, cmap='tab10')
    axs[1].set_title('LDA 2D Visualization')
    plt.colorbar(scatter_lda, ax=axs[1])

    plt.tight_layout()
    plt.show()


# %%
# Estimate Gaussians from PCA/LDA
def fit_gaussians(X, y):
    gaussians = []
    for i in range(10):
        X_i = X[y == i]
        if len(X_i) > 1:
            mean = np.mean(X_i, axis=0)
            cov = np.cov(X_i, rowvar=False)
        else:
            mean = X_i.mean(axis=0)
            cov = np.eye(X.shape[1]) * 1e-6  # Fallback for single sample
        gaussians.append(norm(mean=mean, cov=cov))
    return gaussians


gaussians_pca = fit_gaussians(train_pca, train_targets)
gaussians_lda = fit_gaussians(train_lda, train_targets)


# %%
# Compute predictions
def predict(gaussians, X):
    n_samples = X.shape[0]
    n_classes = len(gaussians)
    log_likelihoods = np.zeros((n_samples, n_classes))
    for i, g in enumerate(gaussians):
        log_likelihoods[:, i] = g.logpdf(X)
    return np.argmax(log_likelihoods, axis=1)


pred_pca = predict(gaussians_pca, test_pca)
pred_lda = predict(gaussians_lda, test_lda)


# Compute accuracy
acc_pca = np.mean(pred_pca == test_targets)
acc_lda = np.mean(pred_lda == test_targets)
print(f"PCA Accuracy: {acc_pca:.4f}")
print(f"LDA Accuracy: {acc_lda:.4f}")


# %%
# Compute the confusion matrices for PCA and LDA
cm_pca = confusion_matrix(test_targets, pred_pca)
cm_lda = confusion_matrix(test_targets, pred_lda)

# Plot Confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay(cm_pca, display_labels=range(10)).plot(ax=axs[0], cmap='Blues')
axs[0].set_title('PCA Confusion Matrix')
ConfusionMatrixDisplay(cm_lda, display_labels=range(10)).plot(ax=axs[1], cmap='Blues')
axs[1].set_title('LDA Confusion Matrix')
plt.tight_layout()
plt.show()


# - RESULTS:
# PCA accuracy: 0.8779
# LDA accuracy: 0.8950
