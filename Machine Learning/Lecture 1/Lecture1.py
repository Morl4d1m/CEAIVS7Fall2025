# ###################################
# Group ID : 720
# Members : Aksel Porsborg Andersen, Magnus Stensli, Takekazu Gade Nakabayashi, Paula Lopez Urquia, Christian Lykke Jørgensen, Nusrat Nabi
# Date : 05/09/2025
# Lecture: Lecture 1 Introduction to Machine Learning 
# Dependencies: numpy, matplotlib, sklearn
# Python version: 3.13.7
# Functionality: 
# ###################################

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# %%
# Load data
work_dir = os.getcwd() # CHANGE THIS WITH YOUR CURRENT DIR!
save_path = work_dir + r'\ex01_data.npy'

data = np.load(save_path) # TODO: load data (check https://numpy.org/doc/stable/reference/generated/numpy.load.html)
    
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

# %%
# Sample data and plot
np.random.seed(100)

sample_points = 10
sample_idx = np.sort(np.random.choice(len(x1), sample_points))

x1_sub = np.take(x1, sample_idx)
x2_sub = np.take(x2, sample_idx) # TODO: sample x2

plt.figure()
plt.scatter(x1_sub, x2_sub, label='Sampled data')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()

# %%
def polyfit(x1, x2, deg, regularization=0, y = None, show_sums_of_squares=False):
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=deg)
    x1_poly = poly.fit_transform(x1[:, np.newaxis])
    
    # Create the Ridge regression model
    ridge_reg = Ridge(alpha=regularization)
    
    # Fit the model
    ridge_reg.fit(x1_poly, x2) # TODO: check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for further information
    
    # Generate x values for the regression line/curve
    x_plot = np.linspace(np.min(x1), np.max(x1), 1000) # TODO: Generate 1000 samples between the minimum and maximum values of x1
    x1_poly_plot = poly.transform(x_plot[:, np.newaxis])
    
    # Predict y_hat values
    y_hat_plot = ridge_reg.predict(x1_poly_plot) # TODO: Compute predictions using the 1000 samples you generated
    y_hat = ridge_reg.predict(x1_poly) # TODO: Use the ridge regression to predict x2 given x1
    
    # Extract coefficients
    coefs = ridge_reg.intercept_, *ridge_reg.coef_[1:]
    
    # Here we calculate mean-square-error (MSE)
    MSE = np.mean((x2-y_hat)**2) # TODO: implement MSE equation

    # At last we plot the observation and the regression line/curve
    plt.figure()
    
    # The following 2 lines is only to show the "squares" to be minimized
    if show_sums_of_squares:
        plt.plot((x1, x1), (x2, ridge_reg.predict(x1_poly)), 
                 color='limegreen')
    
    plt.scatter(x1, x2, label='Observations')
    plt.plot(x_plot, y_hat_plot, '-', label='Polynimial fit of degree %i' % deg, color='orange')
    if y is not None:
        plt.plot(np.linspace(-5,5,500), y, label='Ground truth', color='red')
    
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim([-20, 40])
    plt.legend()
    plt.grid()
    plt.show()
    
    return coefs, MSE

# %%
# 1) Experiment with different polynomial degrees:
#deg = 5
print('Exercise 1')
for deg in [0, 1, 2, 3, 4, 9]:
    coefs, mse = polyfit(x1_sub, x2_sub, deg=deg, y=y, show_sums_of_squares=True)
    print('MSE = ', round(mse,3))

# %%
# 2) Explore L2 regularization:
print('Exercise 2')
deg = 9
for reg in [1.01, 10.001, 100.0001]:
    coefs, mse = polyfit(x1_sub, x2_sub, deg=deg, regularization=reg, y=y, show_sums_of_squares=True) # TODO
    print('MSE = ', round(mse,3)) 

# %%
# 3) Investigate the effect of sample size.
print('Exercise 3')
np.random.seed(100)

sample_points = 100
sample_idx = np.sort(np.random.choice(len(x1), sample_points, replace=False)) # TODO

x1_sub_100 = np.take(x1, sample_idx) # TODO
x2_sub_100 = np.take(x2, sample_idx) # TODO


deg = 9
reg = 1e-2
coefs, mse = polyfit(x1_sub_100, x2_sub_100, deg=deg, regularization=reg, y=y)
print('MSE = ', round(mse,3)) 

