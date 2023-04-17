# My Utility : auxiliars functions

import pandas as pd
import numpy as np


# load config param

def load_cnf(ruta_archivo='cnf.csv'):

    with open(ruta_archivo, 'r') as archivo_csv:

        conf = [int(i) if '.' not in i else float(i)
                for i in archivo_csv if i != '\n']

    return conf


# Initialize weights for SNN-SGDM
def iniWs(inshape, Param):

    in_shape, out_shape, layer1_node, layer2_node = inshape, Param[0], Param[4], Param[5]

    W1 = iniW(layer1_node, in_shape)
    W2 = iniW(layer2_node, layer1_node)
    W3 = iniW(out_shape, layer2_node)
    W = list((W1, W2, W3))

    V = []
    for i in range(len(W)):
        V.append(np.zeros(W[i].shape))

    return W, V


# Initialize weights for one-layer

def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return w

# Feed-forward of SNN


def forward(X, W, Param):
    Act_M = []  # data de cada muestra

    for index, x in X.iterrows():

        x = np.asarray(x.T).reshape(-1, 1)
        A = []
        z = []
        Act = []

        # data input
        z.append(x)
        A.append(x)
        # primera capa
        x = np.dot(W[0], x)
        z.append(x)

        x = act_function(x, act=Param[6])
        A.append(x)
        # segunda capa

        x = np.dot(W[1], x)
        z.append(x)

        x = act_function(x, act=Param[6])
        A.append(x)

        # salida
        x = np.dot(W[2], x)
        z.append(x)

        y = act_function(x, act=4)  # siempre sigmoid
        A.append(y)

        Act.append(A)
        Act.append(z)

        Act_M.append(Act)

    return Act_M


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


def gradW(Act_M, Ye, W, Param):
    '''
    Act = lista de resultados de cada capa,
    data activada en [0] y no activada en [1]
    '''

    Cost = 0
    gW_M = []
    for i, Act in enumerate(Act_M):
        L = len(Act[0])-1
        ye = Ye.iloc[[i]].T
        gW = []

        Cost += float((np.sum(np.square(Act[0][L] - ye), axis=0))/2)

        # error salida
        delta = np.multiply(Act[0][L] - ye, deriva_act(Act[1][L], act=4))
        gW_l = np.dot(delta, act_function(Act[0][L-1], act=4).T)

        gW.append(gW_l)

        # error capas ocultas

        for l in reversed(range(L-1)):
            t1 = np.dot(W[l+1].T, delta)

            t2 = deriva_act(Act[1][l+1], act=Param[6])

            delta = np.multiply(t1, t2)

            t3 = act_function(Act[0][l], act=Param[6]).T

            gW_l = np.dot(delta, t3)
            gW.append(gW_l)

        gW.reverse()

        if i == 0:
            gW_M = gW
        else:
            gW_M = [np.add(gW_M[i], gW[i]) for i in range(len(gW))]

    m = len(Act_M)
    Cost = Cost/m
    gW = [gW/m for gW in gW_M]
    return gW, Cost

# Update W and V


def updWV_sgdm(W, V, gW, Param):

    tasa = Param[9]
    beta = Param[10]

    for i in range(len(W)):
        V[i] = (beta * V[i]) + (tasa*gW[i])
        W[i] = W[i] - V[i]

    return W, V

# Measure


def metricas(y, z):

    z = [z[i][0][-1] for i in range(len(z))]
    z = np.asarray(z).squeeze()
    y = np.asarray(y)
    cm, cm_m = confusion_matrix(z, y)
    
    Fsc = []
    for i in range(len(cm_m)):
        TP = cm_m[i,0,0]
        FP = cm_m[i,0,1]
        FN = cm_m[i,1,0]
        TN = cm_m[i,1,1]
        
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        Fsc.append(( 2 * Precision * Recall ) / ( Precision + Recall ))
     
    Fsc.append( sum(Fsc)/len(Fsc) )

    return cm, Fsc

# Confusion matrix

def confusion_matrix(z, y):
    
    m= len(y[0])
    
    y = np.argmax(y, axis=1)
    z = np.argmax(z, axis=1)
    
    cm = np.zeros((m,m))
    for i in range(len(y)):
         cm[z[i] ,y[i]] +=1
    
    cm_m = np.zeros((cm.shape[0], 2, 2)) #matriz confusion por clase

    for i in range(cm.shape[0]):
        cm_m[i,0,0] = cm[i,i] #TP
        cm_m[i,0,1] = np.sum(np.delete(cm[i,:], i, axis=0)) #FP
        cm_m[i,1,0] = np.sum(np.delete(cm[:,i], i, axis=0)) #FN
        cm_m[i,1,1] = np.sum(np.delete(np.delete(cm, i, axis=1),i , axis=0 )) #TN
    
    return cm , cm_m
# -----------------------------------------------------------------------
