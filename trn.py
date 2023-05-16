# SNN's Training :

import pandas     as pd
import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse(W,Cost):
    np.savez('w_snn.npz', W[0],W[1],W[2])
    
    df = pd.DataFrame( Cost )
    df.to_csv('costo.csv',index=False, header = False )
    
    return

#gets Index for n-th miniBatch
def get_Idx_n_Batch(n,M):
    
    Idx = (n*M,(n*M)+M)
    
    return(Idx) #tuple de indices


#miniBatch-SGDM's Training 
def trn_minibatch(X, Y, W, V, Param):    
 
    M = Param[8]
    nBatch = X.shape[1]// M
    Cost = []
    for n in range(nBatch):

        Idx = get_Idx_n_Batch(n,M)
        
        xe = X[:,slice(*Idx)]
        ye = Y[:,slice(*Idx)]
        Act = ut.forward(xe, W , Param)
        gW, Cost = ut.gradW(Act, ye, W, Param)
        
        W , V = ut.updWV_sgdm(W, V, gW, Param)
    
    return Cost, W, V

import random

#sort data random
def sort_data_ramdom(X,Y):

    idx = np.random.permutation(X.index)
    
    return X.reindex(idx), Y.reindex(idx)


#SNN's Training 
def train(X,Y,Param):    
    W,V   = ut.iniWs(X.shape[1],Param)
    
    MSE = []
    X,Y=np.asarray(X.T), np.asarray(Y.T)
    for Iter in range(1,Param[11]+1):
        idx   = np.random.permutation(X.shape[1])
        X,Y = X[:,idx],Y[:,idx]
        
        Cost, W, V = trn_minibatch(X ,Y ,W ,V , Param)
        MSE.append(np.mean(Cost))
        if Iter%10 == 0:
            print('Iterar-SGD: ', Iter,' ', MSE[Iter-1])
    
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

