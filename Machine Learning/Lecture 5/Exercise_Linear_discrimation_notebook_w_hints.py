# %%
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB

# %%
def create_complete_datasets(data_dict):
    '''
    Function for creating complete training and test sets containing
    all classes.
    '''
    #Empty list
    trainset = []
    traintargets =[]
    testset = []
    testtargets =[]
    
    #For each class
    for i in range(10):
        trainset.append(data_dict["train%d"%i])
        traintargets.append(np.full(len(data_dict["train%d"%i]),i))
        testset.append(data_dict["test%d"%i])
        testtargets.append(np.full(len(data_dict["test%d"%i]),i))
    
    #Concatenate into to complete datasets
    trainset = np.concatenate(trainset)
    traintargets = np.concatenate(traintargets)
    testset = np.concatenate(testset)
    testtargets = np.concatenate(testtargets)
    return trainset, traintargets, testset, testtargets

file = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Machine Learning\Lecture 5\mnist_all.mat"
data = loadmat(file)

#Complete training and test sets
train_set, train_targets, test_set, test_targets = create_complete_datasets(data)

# %%
n_components = 9

#PCA
pca=PCA(n_components=n_components)
xTrainPCA=pca.fit_transform(train_set)
xTestPCA=pca.transform(test_set)

#print(xTrainPCA)
#print(xTestPCA)

#LDA
lda=LDA(n_components=n_components)
xTrainLDA=lda.fit_transform(train_set,train_targets)
xTestLDA=lda.transform(test_set)

#print(xTrainLDA)
#print(xTestLDA)

# %%
# Analyze proportion of Variance. If num_components=2 try to visualize dim. reduced data.


# %%
# Estimate Gaussians from PCA/LDA

GaussPCA=GaussianNB()
PCAFit=GaussPCA.fit(xTrainPCA,train_targets)
GaussLDA=GaussianNB()
LDAFit=GaussLDA.fit(xTrainLDA,train_targets)

# %%
#Compute predictions
PCAPrediction=GaussPCA.predict(xTestPCA)
LDAPrediction=GaussLDA.predict(xTestLDA)

#Compute accuracy
PCAPrecision=np.mean(PCAPrediction == test_targets)
print(PCAPrecision)
LDAPrecision=np.mean(LDAPrediction == test_targets)
print(LDAPrecision)

#However, for future reference, if only the accuracy is desired, this function is faster to call:
PCAAccuracy=GaussPCA.score(xTestPCA, test_targets, sample_weight=None)
print(PCAAccuracy)
LDAAccuracy=GaussLDA.score(xTestLDA, test_targets, sample_weight=None)
print(LDAAccuracy)

# %%
#Compute the confusion matrices for PCA and LDA


#Plot Confusion matrices



