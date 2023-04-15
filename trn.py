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
    Cost = 0
    
    for n in range(nBatch):
        
        Idx = get_Idx_n_Batch(n,M)
        xe = X[slice(*Idx)]
        ye = Y[slice(*Idx)]
        Act = ut.forward(xe , W , Param)
        gW, Cost = ut.gradW(Act, ye, W, Param)
        W , V = ut.upd_WV_sgdm(W, V, gW, Param)
    
    return Cost, W, V

import random

#sort data random
def sort_data_ramdom(X,Y):
    
    zipped = list(zip(X, Y))
    random.shuffle(zipped)
    X, Y = zip(*zipped)
    
    return X, Y


#SNN's Training 
def train(X,Y,Param):    
    W,V   = ut.iniWs(Param)
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
    df = pd.read_csv(ruta_archivo)
    X = df['X'].values.tolist()
    Y = df['Y'].values.tolist()
    
    return X, Y
    
   
# Beginning ...
def main():
    param       = ut.load_cnf()            
    xe,ye       = load_data_trn()   
    W,Cost      = train(xe,ye,param)             
    save_w_mse(W,Cost)
       
if __name__ == '__main__':   
	 main()

