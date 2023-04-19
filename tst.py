import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm,Fsc):
    
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv('cmatriz.csv', index=False, header = False)
    
    df_Fsc = pd.DataFrame(Fsc)
    df_Fsc.to_csv('fscores.csv', index=False, header = False)
    
    return()

def load_w():
    ws = np.load('w_snn.npz')
    return [ws[i] for i in ws.files]


# Load data to test the SNN
def load_data_test(ruta_archivo='dtrain.csv'):

    df = pd.read_csv(ruta_archivo, converters={'COLUMN_NAME': pd.eval})
    X = df.filter(regex='x_')
    Y = df.filter(regex='y_')
        
    return X, Y
   

# Beginning ...
def main():			
    param  = ut.load_cnf()
    xv,yv  = load_data_test()
    W      = load_w()
    zv     = ut.forward(xv,W,param)
    #print(zv)
    cm,Fsc = ut.metricas(yv,zv) 	
    save_measure(cm,Fsc)
		

if __name__ == '__main__':   
	 main()

