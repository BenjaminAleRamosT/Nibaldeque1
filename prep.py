import pandas as pd
import numpy as np
import utility as ut

import math
import matplotlib.pyplot  as plt

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

#Discrete Fourier Transform
def DFT(x, k):
    #se cambio el output para no dividirlo por N
    suma = 0
    N = len(x)
    for n in range(N):
        suma += x[n] * np.exp( - (1j * ((2*math.pi)/N)*k*n ) )
    return suma

#uso de la DFT
x = [1,2,3,4,5,6,7,8,9,10,11,12,13]
x = [complex(x_i) for x_i in x]
i_x = [ DFT(x, i) for i in range(len(x)) ]

#Inverse Fourier Transform
def IDFT(x, n):
    #el output se dividio por N en vez de multiplicar
    suma = 0
    N = len(x)
    for k in range(N):
        suma += x[k]  * np.exp( 1j * ((2*math.pi)/N)*k*n )
    return suma/N

#uso de la IDFT 
a = [ IDFT(i_x, i) for i in range(len(i_x)) ]


def amplitud_espectral(x):
    A = x.real
    B = x.imag
    return np.sqrt( np.sum([np.square(A) , np.square(B)] ,axis = 0) )

# sampling rate
sr = 128
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7   
x += 0.5* np.sin(2*np.pi*freq*t)

plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')

plt.show()

chirp_x = [ DFT(x, i) for i in range(len(x)) ]

plt.plot(x)
plt.show()
plt.stem( A(np.asarray(chirp_x)), 'b', \
         markerfmt=" ", basefmt="-b" )
plt.show()

# Fourier spectral entropy
def entropy_spectral():
    
    
    

    
    
    return

# Hankel-SVD


def hankel_svd(X, nFrame, lFrame):

    N = nFrame
    L = lFrame
    K = N - L + 1
    
    H = np.empty((0,K), int)
    
    for n in range(L):
        H = np.vstack(( H , X[n:n+K] ))
    
    U, S, V = np.linalg.svd(H)
    V= V.T
    
    C = []
    for i in range(S.shape[0]):
        aux = np.array([S[i] * U[:,i]]).T
        H_i = np.dot(aux, [V[:,i]])
        
        C_i = np.hstack((H_i[0,:] , H_i[1:,-1]))  #la primera fila y la ultima columna (-primer elemento)
        C.append(C_i)
        
    C = np.asarray(C)
    
    X_new = np.sum(C, axis = 0 )
    
    U, Svalues_C, V = np.linalg.svd(C)
    
    return Svalues_C

##Para probar
#X = [3.5186, 3.2710, 1.0429, 2.3774, 0.0901, 1.7010, 1.2509, 0.6459]
#nFrame, lFrame = 8,3
#hankel_svd(X, nFrame, lFrame)

# Hankel's Diadica

def hankel_diadica(X):
       
    if len(X) < 3 :
        print('No es posible realizar la descomposicion con menos de 3 valores')
        return
    
    H = np.empty((0,len(X)-1), int)
    for n in range(2):
        H = np.vstack(( H , X[n:n+len(X)-1] ))
    U, S, V = np.linalg.svd(H)
    V= V.T
    
    C = []
    for i in range(2):
        aux = np.array([S[i] * U[:,i]]).T
        H_i = np.dot(aux, [V[:,i]])
        
        C_i = []
        C_i.append(H_i[0,0]) 
        for j in range(len(X)-2):
            c = (H_i[1,j] + H_i[0,j+1])/2
            C_i.append(c)
        C_i.append(H_i[-1,-1])
        C_i = np.asarray(C_i)
        C.append(C_i)
    return C


#gets Index for n-th Frame
def get_Idx_n_Frame(n,l):
    
    Idx = (n*l,(n*l)+l)
    
    return(Idx) #tuple de indices



# Hankel's features

def hankel_features(X,Param):
    
    nFrame = Param[1]
    lFrame = Param[2]
    jNiveles = Param[3]
    
    F = []
    
    for n in range(nFrame):
        
        #se puede hacer uso de la misma funcion de los batch para los frames
        Idx = get_Idx_n_Frame(n,lFrame)
        x_frame = X[slice(*Idx)]
        
        #para evitar errores al descomponer se ignora el frame si tiene menos de 2 elementos
        if len(x_frame) > 2:
            C = []
            C.append(x_frame)
            
            for j in range(jNiveles):
                C_j = []
                for item in C:
                    C_j.extend( hankel_diadica(item) )
                C = C_j
            
            e = []
            U, S, V = np.linalg.svd(np.asarray(C))
            for item in C:
                e.append(entropy_spectral(item))
            np.asarray(e)
            
        F.append( np.hstack(( e , S )) )
                
    return F
    
X = [1,2,3,4,5,6,7,8]
hankel_features(X, (3,3,3,3) )


# Obtain j-th variables of the i-th class
def data_class(x, j, i):
    
    x.iloc[: , -1]
    
    return


# Create Features from Data
def create_features(X, Param):
    
    for i in range(Param[0]):
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
    
    data = []
    
    for n in range(n_class):
        
        path_csv = path + '\class'+str(n+1)+'.csv'
        data_class = np.genfromtxt(path_csv, delimiter=',')
        
        ##añadir target aca o en otro lado
        #feat = np.ones(data_class.shape[1]) * n # los datos son las columnas
        #data_class = np.vstack((data_class,feat))
        #print(data_class.shape)
        
        data.append(data_class)
        
    return(data)

# Parameters for pre-proc.


# Beginning ...
def main():
    Param = ut.load_cnf('cnf.csv')
    Data = load_data(Param)
    #InputDat, OutDat = create_features(Data, Param)
    #InputDat = data_norm(InputDat)
    #save_data(InputDat, OutDat)


if __name__ == '__main__':
    main()
