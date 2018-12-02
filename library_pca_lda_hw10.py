# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This library contains all the functions used in PCA and LDA tasks
# of Homework 10 of ECE661.

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

zero_thresh = 1e-15  # absolute value below which eigen value treated as 0

class image_vector():
    # Class of image vectors
    # Parameters in class:
    #   1. self.pixels
    #   2. self.label
    def __init__(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(image.shape)
        # self.ch1 = image[:, :, 0]
        # b, g, r = cv2.split(image)
        self.pixels = image.flatten()
        self.pixels = self.pixels / np.linalg.norm(self.pixels)  # normalize
        text = filename.split('_')[-2]
        text = text.split('\\')
        self.label = text[-1]
        # print(filename, self.label)
        

def load_data(folder):
    # Loads data from folder
    data = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            # print(os.path.join(root, name))
            filename = os.path.join(root, name)
            data.append(image_vector(filename))
    # Sort the data based on their label
    data = sorted(data, key = lambda x:int(x.label))
    return data


def find_mean(data):
    # Finds the mean of a list of 'image_vector's
    image_vecs = [data[i].pixels for i in range(len(data))]
    mean_pixels = sum(image_vecs) / len(data)
    return mean_pixels


def generate_X(data):
    # Generates the X matrix from data for PCA
    # Returns X and mean
    X = np.empty((data[0].pixels.shape[0], len(data)))
    for i in range(len(data)):
        X[:, i] = data[i].pixels
    X = normalize_matrix(X)
    m = np.mean(X, axis = 1)
    m = m.reshape((-1, 1))
    # print(X.shape, m.shape)
    # X = X - m
    return [X, m]


def normalize_matrix(X):
    # Normalizes the columns of the X
    for i in range(X.shape[1]):
        column = X[:, i]
        # print(column)
        X[:, i] = column / np.linalg.norm(column)
    return X


def eigen_decompose(X):
    # Carries out eigen decomposition of XT*A and returns the eigen
    # values and eigen vectors.
    A = np.matmul(X.T, X)
    lambdas, us = np.linalg.eig(A)
    ws = np.matmul(X, us)
    return [lambdas, ws]
    

def pca_task(train_folder, test_folder, p_max):
    # Carries out the PCA based task
    # p_max specifies the maximum subspace dimensionality p required in plot
    data_train = load_data(train_folder)
    # Generate Xi and m for all train data
    [X_train, m_train] = generate_X(data_train)
    [lambdas, ws] = eigen_decompose(X_train - m_train)
    ws = normalize_matrix(ws)
    
    # Generate Xi for test data
    data_test = load_data(test_folder)
    [X_test, _] = generate_X(data_test)
    # print(lambdas.shape)
    # print(ws.shape)
    pca_accuracy = []
    for p in range(1, p_max+1):
        print('Computing accuracy for p = {} of {}'.format(p, p_max))
        # print(ws.shape, X)
        WK = ws[:, :p]
        y_train = get_coordinates(X_train, WK, m_train)
        y_test = get_coordinates(X_test, WK, m_train)
        
        # Finds the indices of the nearest neighbors of the test data in 
        # train data
        classify_indices = classify(y_train, y_test)
        
        # Find the labels of the nearest neighbors for each test sample
        classify_labels = [data_train[i].label for i in classify_indices]
        [accuracy, match] = classification_error(classify_labels, data_test)
        pca_accuracy.append(accuracy)
        print('Accuracy = {:10.7f}'.format(accuracy))
        # print(match)
    fig = plt.figure()
    plt.plot(list(range(1, p_max+1)), pca_accuracy, color='red', \
             marker='x', markersize = 3)
    return [list(range(1, p_max+1)), pca_accuracy]
    
    
def lda_task(train_folder, test_folder, p_max):
    # Carries out the LDA based task
    # p_max specifies the maximum subspace dimensionality p required in plot
    data_train = load_data(train_folder)
    
    # Find U and Z
    [U, Z] = get_ws_lda(data_train)
    print('U and Z computed')
    
    # Generate Xi and m for all train data
    [X_train, m_train] = generate_X(data_train)
    
    # Generate Xi for test data
    data_test = load_data(test_folder)
    [X_test, _] = generate_X(data_test)
    
    lda_accuracy = []
    for p in range(1, p_max+1):
        print('Computing accuracy for p = {} of {}'.format(p, p_max))
        
        # Compute WP
        U_cap = U[:, -p:]
        WPT = np.matmul(U_cap.T, Z.T)
        WP = WPT.T
        WP = normalize_matrix(WP)
        
        y_train = get_coordinates(X_train, WP, m_train)
        y_test = get_coordinates(X_test, WP, m_train)
        
        # Finds the indices of the nearest neighbors of the test data in 
        # train data
        classify_indices = classify(y_train, y_test)
        
        # Find the labels of the nearest neighbors for each test sample
        classify_labels = [data_train[i].label for i in classify_indices]
        [accuracy, match] = classification_error(classify_labels, data_test)
        lda_accuracy.append(accuracy)
        print('Accuracy = {:10.7f}'.format(accuracy))
        # print(match)
    fig = plt.figure()
    plt.plot(list(range(1, p_max+1)), lda_accuracy, color='blue', \
             marker='x', markersize = 3)
    return [list(range(1, p_max+1)), lda_accuracy]
    
    
def get_ws_lda(data):
    # Finds the Ws / basis vectors for LDA
    class_vectors = np.empty((data[0].pixels.shape[0], 0))
    classes = []
    oldlabel = ''
    
    # Segregate the data into classes
    for sample in data:
        label = sample.label
        if label == oldlabel:
            pixels = sample.pixels.reshape((-1, 1))
            # print(class_vectors.shape, pixels.shape)
            class_vectors = np.concatenate((class_vectors, pixels),\
                                           axis = 1)
        else:
            oldlabel = label
            classes.append(class_vectors)
            class_vectors = sample.pixels.reshape((-1, 1))
    classes.append(class_vectors)
    classes.pop(0)
    
    # Find the class means and X_w
    class_means = []
    X_w = np.empty((data[0].pixels.shape[0], 0))
    for i in range(len(classes)):
        class_vectors = classes[i]
        class_vectors = normalize_matrix(class_vectors)
        classes[i] = class_vectors
        class_mean = np.mean(class_vectors, axis = 1)
        class_means.append(class_mean)
        X_w = np.concatenate((X_w, class_vectors - \
                              class_mean.reshape((-1, 1))), axis = 1)
    
    # Generate pseudo S_B
    global_mean = find_mean(data)
    
    print('Computing S_B')
    Mi = generate_X_LDA(class_means)
    M = Mi - global_mean.reshape((-1, 1))
    sb_pseudo = np.matmul(M.T, M) / len(classes)
    lambdas, us = sorted_eigen(sb_pseudo)
    # print(lambdas)
    Y = np.matmul(M, us)
    
    # Z = Y D_B^(0.5)
    # print(lambdas)
    # print(lambdas ** (-0.5))
    invsqrt_DB = np.diag(lambdas ** (-0.5))
    Z = np.matmul(Y, invsqrt_DB)
    # print(invsqrt_DB.sum())
    
    # Generate pseudo S_w
    # print(X_w.shape)
    vect = np.matmul(Z.T, X_w)
    pseudo_vect = np.matmul(vect.T, vect)
    # print(pseudo_vect.sum())
    lambdas, U = sorted_eigen_imag(pseudo_vect)
    # print(U)
    U = np.matmul(vect, U)
    # print(U)
    
    # Return U, Z
    return [U, Z]


def sorted_eigen_imag(A):
    # Returns the sorted eigen values and eigen vectors in decreasing order
        
    eigenValues, eigenVectors = np.linalg.eigh(A)

    idx = eigenValues.argsort()[::-1]
    
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    return [eigenValues, eigenVectors]


def sorted_eigen(A):
    # Returns the sorted eigen values and eigen vectors in decreasing order
    eigenValues, eigenVectors = np.linalg.eigh(A)

    idx = eigenValues.argsort()[::-1]
    # print(eigenValues.shape, eigenVectors.shape)
    
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    for i in range(len(eigenValues)):
        if abs(eigenValues[i]) < zero_thresh:
            eigenValues[i] = 0
    
    index = np.where(eigenValues == 0)[0][0]
    eigenValues = eigenValues[:index]
    eigenVectors = eigenVectors[:, :index]
    # print(eigenValues.shape, eigenVectors.shape)
    
    return [eigenValues, eigenVectors]


def generate_X_LDA(data):
    # Generates the X matrix from data (list) for LDA
    # Returns X
    X = np.empty((data[0].shape[0], len(data)))
    for i in range(len(data)):
        X[:, i] = data[i]
    return X
    
    
def classification_error(labels, data):
    # Computes the classification error using the labels into which 
    # classified, and the data
    match = []
    num_correct = 0
    for i in range(len(labels)):
        label_assigned = labels[i]
        label_real = data[i].label
        # print(label_assigned, label_real)
        if label_assigned == label_real:
            match.append(True)
            num_correct += 1
        else:
            match.append(False)
    accuracy = num_correct/len(labels) * 100
    return [accuracy, match]
    
    
def classify(y_train, y_test):
    # Classify the test data based on training data
    classification_matrix = []
    for i in range(y_test.shape[1]):
        test_sample = y_test[:, i]
        classification_matrix.append(nearest_neigh(test_sample, y_train)[0])
    return classification_matrix
        
        
def nearest_neigh(sample, all_data):
    # Finds the nearest neighbor of sample in all_data
    # Return the index of the nearest neighbor, and the smallest dist
    dist = np.Inf
    nearest = []
    for i in range(all_data.shape[1]):
        if np.linalg.norm(sample - all_data[:, i]) < dist:
            nearest = i
            dist = np.linalg.norm(sample - all_data[:, i])
    return [nearest, dist]
    
    
def get_coordinates(X, WK, m):
    # Get the coordinates of the image vectors in X (columns) by
    # projecting them onto WK, using the mean image vector m
    X = normalize_matrix(X)
    y = np.matmul(WK.T, X-m)
    return y


def fraction_power(A, alpha):
    # This function is not used.
    # Find the fractional power of matrix A, i.e. A^alpha
    lambdas, u = np.linalg.eig(A)
    D = np.diag(lambdas ** alpha)
    A_alpha = np.matmul(u, np.matmul(D, np.linalg.inv(u)))
    return A_alpha