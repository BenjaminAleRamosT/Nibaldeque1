import pandas as pd
import numpy as np
import utility as ut

# Save Data from  Hankel's features


def save_data(X, Y):

    return

# normalize data


def data_norm(x, a = 0.01, b = 0.99):

    x_max = np.max(x)
    x_min = np.min(x)
    x = ( ( ( x - x_min )/( x_max - x_min ) ) * ( b - a ) ) + a
    
    return x

# Binary Label


def binary_label(i):

    #cantidad de clases distintas
    n_class = len(np.unique(i))
    
    
    #se genera una matriz de 0s, 
    #luego se le suma 1 en las posiciones que corresponda
    #para la clase 0 se le suma un 1 en la posicion 0 de la fila Idx
    
    n_muestras = len(i)
    label = np.zeros( ( n_muestras ,n_class ))
    
    for Idx in range(n_muestras):
        label[Idx,i[Idx]] =+ 1 

    return label


# Fourier spectral entropy
def entropy_spectral():

    return

# Hankel-SVD


def hankel_svd():

    return

# Hankel's features


def hankel_features(X, nFrame, lFrame):
    
    #recordatorio parametros
    #Línea 2: Número de Frame : 50
    #Línea 3: Tamaño del Frame : 256
    #Línea 4: Nivel de Descomposición : 3    

    N = nFrame
    L = lFrame
    K = N - L + 1
    
    H = np.empty((0,K), int)
    
    for n in range(L):
        H = np.vstack(( H , X[n:n+K] ))
    
    U, S, V = np.linalg.svd(H)
    
    C = []
    
    for i in range(S.shape[0]):
    
        aux = np.array([S[i] * U[:,i]]).T
        H_i = np.dot(aux, [V[i,:]])
        
        C_i = np.hstack((H_i[0,:] , H_i[1:,-1]))  #la primera fila y la ultima columna (-primer elemento)
        
        C.append(C_i)
        
        #print(C_i)
        #print(H_i)
    
    C = np.asarray(C)
    
    X_new = np.sum(C, axis = 0 )
    
    #print(X_new) #funciona
    
    Svalues_C = np.linalg.svd(C)
    
    return Svalues_C

X = [3.5186, 3.2710, 1.0429, 2.3774, 0.0901, 1.7010, 1.2509, 0.6459]
nFrame, lFrame = 8,3

hankel_features(X, nFrame, lFrame)
# Obtain j-th variables of the i-th class
def data_class(x, j, i):
    
    x.iloc[: , -1]
    
    return


# Create Features from Data
def create_features(X, Param):
    
    for i in range(n_class):
        for j in range(n_var):
            X = data_class()
            F = hankel_features(X, Param)
            datF = apilar_features()
    
    return


# Load data from ClassXX.csv

import os

def load_data(Param):
    n_class = Param[0] #add [0]
    
    path = 'DATA\Data' + str(n_class)
    
    for n in range(n_class):
        
        path_csv = path + '\class'+str(n+1)+'.csv'
        
        print(path_csv)
        
        data = np.genfromtxt(path_csv, delimiter=',')
        
    return()


# Parameters for pre-proc.


# Beginning ...
def main():
    Param = ut.load_cnf('cnf.csv')
    Data = load_data()
    InputDat, OutDat = create_features(Data, Param)
    InputDat = data_norm(InputDat)
    save_data(InputDat, OutDat)


if __name__ == '__main__':
    main()
