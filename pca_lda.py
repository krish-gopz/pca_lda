# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This script carries out the PCA and LDA tasks for Homework 10 of ECE661

import library_pca_lda_hw10 as lib
import matplotlib.pyplot as plt
import numpy as np
dpi_value = 600

# Compute the accuracies for PCA
[x_pca, y_pca] = lib.pca_task('train', 'test', 30)

# Compute the accuracies for LDA
[x_lda, y_lda] = lib.lda_task('train', 'test', 30)

# Plot and save the figure showing comparison
fig = plt.figure()
plt.plot(x_pca, y_pca, color='blue', marker='x', markersize = 5, \
         label = 'PCA')
plt.plot(x_lda, y_lda, color='red', marker='x', markersize = 5, \
         label = 'LDA')
plt.xlabel('Subspace dimension')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('PCA_LDA.png', dpi = dpi_value)
plt.close()

# Save the values of accuracy w.r.t the subspace dimensions in txt file
x_pca = np.array(x_pca)
y_pca = np.array(y_pca)
x_lda = np.array(x_lda)
y_lda = np.array(y_lda)
x_pca = x_pca.reshape((-1, 1))
y_pca = y_pca.reshape((-1, 1))
x_lda = x_lda.reshape((-1, 1))
y_lda = y_lda.reshape((-1, 1))
X = np.concatenate((x_pca, y_pca, x_lda, y_lda), axis = 1)
np.savetxt('PCA_LDA.txt', X, fmt='%10.5f')