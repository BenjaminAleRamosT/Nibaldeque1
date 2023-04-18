import pandas as pd
import numpy as np
import utility as ut

import math

# Save Data from  Hankel's features


def save_data(X, Y, Param):
    
    df_x = pd.DataFrame.from_records(X)
    df_x = df_x.add_prefix('x_')
    
    
    df_y = pd.DataFrame.from_records(Y)
    df_y = df_y.add_prefix('y_')
    
    df = pd.concat([df_x, df_y], axis=1)
    
    p = Param[7]
    
    df_train = df.sample(frac=p)
    df_test = df.drop(df_train.index)
    
    df_train.to_csv('dtrain.csv')
    df_test.to_csv('dtest.csv')

    return 

# normalize data


def data_norm(x, a = 0.01, b = 0.99):

    x_max = np.max(x)
    x_min = np.min(x)
    x = ( ( ( x - x_min )/( x_max - x_min ) ) * ( b - a ) ) + a
    
    return x

# Binary Label


def binary_label(i,Param):

    #cantidad de clases distintas
    n_class = Param[0]
   
    label = np.zeros( n_class )
    label[i] = 1
    
    return label

#Discrete Fourier Transform
def DFT(x):
    #se cambio el output para no dividirlo por N
    suma = 0
    N = len(x)
    #x = [complex(x_i) for x_i in x]
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    exp = np.exp( - (1j * ((2*math.pi)/N)*k*n ) )
    
    X = np.dot(exp, x)
    
    return X



#Inverse Fourier Transform
def IDFT(x, n):
    #el output se dividio por N en vez de multiplicar
    suma = 0
    N = len(x)
    for k in range(N):
        suma += x[k]  * np.exp( 1j * ((2*math.pi)/N)*k*n )
    return suma/N

#uso de la IDFT 
#a = [ IDFT(i_x, i) for i in range(len(i_x)) ]


def amplitud_espectral(x):
    A = x.real
    B = x.imag
    return np.sqrt( np.sum([np.square(A) , np.square(B)] ,axis = 0) )

# Fourier spectral entropy
def entropy_spectral(x):
    x = data_norm(np.abs(x)) 
    
    N = len(x)
    
    I = math.ceil(math.sqrt(N))
    
    Xmax ,Xmin = np.max(x), np.min(x)
    l = (Xmax - Xmin)/(I-1) 
    
    p = []
    for i in range(I):
        p_i = len([a for a in x if i*l <= a <= (i*l)+l])/N 
        if p_i != 0:
            p.append(p_i)
       
    p = np.asarray(p)
    return -np.sum( p * np.log2(p)) /np.log2(I)
 
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
    
    #X_new = np.sum(C, axis = 0 )
    
    U, Svalues_C, V = np.linalg.svd(C)
    
    return Svalues_C

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
    
    Idx = (n*l , (n*l) +l)
    
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
                x = DFT(item)
                
                e.append(entropy_spectral(x[:len(x)//2]))
            np.asarray(e)
            
        
        
        F.append( np.hstack(( e , S )) )
                
    return F
    

# Obtain j-th variables of the i-th class
def data_class(Dat, j, i):
    return Dat[i][:,j]


# Create Features from Data
def create_features(Dat, Param):
    # Dat : lista de dataframes de 4 columnas
    #print(len(Dat))
    Y, X = [],[]
    for i in range(Param[0]):
        datF = []
        for j in range(Dat[0].shape[1]):
            print('\tClase: ',i+1,' dato: ',j+1)
            X_dat = data_class(Dat,j,i)
            F = hankel_features(X_dat, Param) # cada F es una muestra o cada datF
            datF.extend(F)
            
            
        Label = binary_label(i,Param)
        for z in range( len(datF) ):
            Y.append(Label)
        X.extend(datF)

    return X, Y


# Load data from ClassXX.csv

def load_data(Param):
    n_class = Param[0] #add [0]
    
    path = 'DATA\Data' + str(n_class)
    
    data = []
    
    for n in range(n_class):
        
        path_csv = path + '\class'+str(n+1)+'.csv'
        data_class = np.genfromtxt(path_csv, delimiter=',')
        data.append(data_class)
        
    return(data)

# Parameters for pre-proc.


# Beginning ...
def main():
    print('Cargando config...')
    Param = ut.load_cnf()
    print('Cargando data...')
    Data = load_data(Param)
    print('Creando features...')
    InputDat, OutDat = create_features(Data, Param)
    print('Normalizando data...')
    InputDat = data_norm(InputDat)
    print('Guardando data...')
    save_data(InputDat, OutDat, Param)
    

if __name__ == '__main__':
    main()
