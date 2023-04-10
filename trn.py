# SNN's Training :

import pandas     as pd
import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse():
    ...
    return

#gets Index for n-th miniBatch
def get_Idx_n_Batch(n,M):
    
    Idx = (n*M,(n*M)+M)
    
    return(Idx) #tuple de indices


#miniBatch-SGDM's Training 
def trn_minibatch(X, Y, W, V, Param):    
 
    M = Param[8]  
 
    nBatch = N/M
    
    for n in range(nBatch):
        
        Idx = get_Idx_n_Batch(n,M)
        xe = X[slice(*Idx)]
        ye = Y[slice(*Idx)]
        Act = forward(xe , W , V, Param)
        gW, Cost = gradW(Act, ye, W, Param)
        W , V = upd_WV_sgdm(W, V, gW, Param)
    
    return(Cost, W, V)

#SNN's Training 
def train(X,Y,Param):    
    W,V   = iniWs()
    MSE = []
    
    for Iter in range(Param[11]):
    
        X,Y = sort_data_ramdom(X,Y)
    
        Cost, W, V = trn_minibatch(X ,Y ,W ,V , Param)
        
        MSE.append(np.mean(Cost))
        if Iter%10 == 0:
            print('Iterar-SGD: ', Iter,' ', MSE(Iter))
    
    return(W,Cost)

# Load data to train the SNN
def load_data_trn():
    
    return()
    
   
# Beginning ...
def main():
    param       = ut.load_cnf()            
    xe,ye       = load_data_trn()   
    W,Cost      = train(xe,ye,param)             
    save_w_cost(W,Cost)
       
if __name__ == '__main__':   
	 main()

