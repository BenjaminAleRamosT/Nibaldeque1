# My Utility : auxiliars functions

import pandas as pd
import numpy as np
import csv

# se puede importar OS

#load config param

def load_cnf(ruta_archivo):
    
    with open(ruta_archivo, 'r') as archivo_csv:
    
        conf = [int(i) if '.' not in i else float(i) for i in archivo_csv if i != '\n']

    return conf


# Initialize weights for SNN-SGDM
def iniWs(Param):
    
    inshape = 2 #PLACEHOLDER que parametro es, deberia ser el largo de las features
    in_shape,out_shape,layer1_node,layer2_node = inshape,Param[0],Param[4],Param[5]
    
    W1 = iniW( layer1_node , in_shape )
    
    W2 = iniW( layer1_node, layer2_node )
    
    W3 = iniW( out_shape , layer2_node )
    
    W = list((W1,W2,W3))
    
    V=[]
    for i in range(len(W)):
        V.append(np.zeros(W[i]))

    return W, V


# Initialize weights for one-layer


def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return w

# Feed-forward of SNN


def forward(x , W , Param):

    A = []
    z = []
    Act = []
    # primera capa

    x = np.dot(x , W[0])
    z.append(x)
    
    x = act_function(x, act=Param[6])
    A.append(x)
    # segunda capa

    x = np.dot(x , W[1])
    z.append(x)
    
    x = act_function(x, act=Param[6])
    A.append(x)

    # salida
    x = np.dot(x * W[2])
    z.append(x)
    
    y = act_function(x, act=4)
    A.append(y)

    Act.append(A)
    Act.append(z)
    
    return Act


# Activation function
def act_function(x, act=0, a_ELU=1, a_SELU=1.6732, lambdd=1.0507):

    # Relu

    if act == 0:
        condition = x > 0
        return np.where(condition, x, np.zeros(x.shape))

    # LRelu

    if act == 1:
        condition = x >= 0
        return np.where(condition, x, x * 0.01)

    # ELU

    if act == 2:
        condition = x > 0
        return np.where(condition, x, a_ELU * np.expm1(x))

    # SELU

    if act == 3:
        condition = x > 0
        return lambdd * np.where(condition, x, a_SELU * np.expm1(x))

    # Sigmoid

    if act == 4:
        return 1 / (1 + np.exp(-1*x))

    return x
# Derivatives of the activation funciton


def deriva_act(x, act=0, a_ELU=1, a_SELU=1.6732, lambd=1.0507):

    # Relu

    if act == 0:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.zeros(x.shape))

    # LRelu

    if act == 1:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.ones(x.shape) * 0.01)

    # ELU

    if act == 2:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), a_ELU * np.exp(x))

    # SELU falta

    if act == 3:
        condition = x > 0
        return lambd * np.where(condition, np.ones(x.shape), a_SELU * np.exp(x))

    # Sigmoid

    if act == 4:
        # pasarle la sigmoid
        return act_function(x, act=4) * (1 - act_function(x, act=4))

    return x

# Feed-Backward of SNN


def gradW(Act, ye, W, Param):
    '''
    Act = lista de resultados de cada capa,
    data activada en [0] y no activada en [1]
    '''
    
    L = len(Act[0])-1
    M = len(ye)
    gW = []
    
    #error salida
    t_e = (np.sum( np.square(Act[0][L] - ye) , axis=1))/2 ##usar ultima capa en Act
    Cost = 1/M * ( np.sum(t_e) )
    
    #como se saca e?, deberia de esr una lista
    delta = np.multiply( Act[0][L] - ye , deriva_act( Act[1][L], act=4 ) )
    gW_l = np.dot( delta, act_function( Act[0][L-1] , act = 4 ).T )
    
    gW.append(gW_l)
    
    #error capa oculta
    for l in range(L-1, 0):
        t1 = np.dot( W[l+1].T , delta )
        
        t2 = deriva_act( Act[1][l] , act = Param[6] )
        
        delta = np.multiply(t1,t2)
        
        t3 = act_function( Act[0][l-1] , act = Param[6] ).T
        
        gW_l = np.dot( delta , t3 )
        gW.append(gW_l)
    
    gW.reverse()
    
    return gW, Cost

# Update W and V


def updWV_sgdm( W , V , gW, Param):
    
    tasa = Param[9]
    beta = Param[10]
    
    for i in range(len(W)):
        V[i] = (beta * V[i]) * (tasa*gW[i])
        W[i] = W[i] - V[i]
    
    return W, V

# Measure


def metricas(x, y):
    
    
    return()

# Confusion matrix


def confusion_matrix(z, y):
    
    cm = ''
    
    return(cm)
# -----------------------------------------------------------------------
