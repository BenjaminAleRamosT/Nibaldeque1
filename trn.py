# SNN's Training :

import pandas     as pd
import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse(W,Cost):
    np.savez('w_snn.npz', W[0],W[1],W[2])
    
    df = pd.DataFrame( Cost )
    df.to_csv('costo.csv',index=False )
    
    return

#gets Index for n-th miniBatch
def get_Idx_n_Batch(n,M):
    
    Idx = (n*M,(n*M)+M)
    
    return(Idx) #tuple de indices


#miniBatch-SGDM's Training 
def trn_minibatch(X, Y, W, V, Param):    
 
    M = Param[8]
    nBatch = len(Y)// M
    Cost = []
    
    for n in range(nBatch):
        Idx = get_Idx_n_Batch(n,M)
        xe = X[slice(*Idx)]
        ye = Y[slice(*Idx)]
        
        
        Cost_minibatch = []
        gW_minibatch = []
        for i in range(len(xe)):
            Act = ut.forward(xe.iloc[[i]] , W , Param)
            gW_n, Cost_n = ut.gradW(Act, ye.iloc[[i]], W, Param)
            
            gW_minibatch.append(gW_n)
            Cost_minibatch.append(Cost_n)
            
            
        #promediar las gradientes y costos
        Cost.append(np.mean(Cost_minibatch))
        #promediar las gradientes de cada muestra
        for i in range(len(gW_minibatch)):
            
        
        
        W , V = ut.updWV_sgdm(W, V, gW, Param)
    
    
    return Cost, W, V

import random

#sort data random
def sort_data_ramdom(X,Y):
    
    #zipped = list(zip(X, Y))
    #random.shuffle(zipped)
    #X, Y = zip(*zipped)
    #X, Y = list(X), list(Y)
    idx = np.random.permutation(X.index)
    
    return X.reindex(idx), Y.reindex(idx)


#SNN's Training 
def train(X,Y,Param):    
    W,V   = ut.iniWs(X.shape[1],Param)
    MSE = []
    
    for Iter in range(Param[11]):
        X,Y = sort_data_ramdom(X,Y)
        Cost, W, V = trn_minibatch(X ,Y ,W ,V , Param)
        
        MSE.append(np.mean(Cost))
        if Iter%10 == 0:
            print('Iterar-SGD: ', Iter,' ', MSE[Iter])
    
    return W,MSE



# Load data to train the SNN
def load_data_trn(ruta_archivo='dtrain.csv'):
    df = pd.read_csv(ruta_archivo, converters={'COLUMN_NAME': pd.eval})
    X = df.filter(regex='x_')
    Y = df.filter(regex='y_')
    
    return X, Y
    
   
# Beginning ...
def main():
    param       = ut.load_cnf()            
    xe,ye       = load_data_trn()
    W,Cost      = train(xe,ye,param)             
    save_w_mse(W,Cost)
       
if __name__ == '__main__':   
	 main()

