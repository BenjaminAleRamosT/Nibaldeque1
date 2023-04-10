import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm,Fsc):
    ...
    return()

def load_w():
    ...
    return(...)



def load_data_test():
    ....
    return(...)
    

# Beginning ...
def main():			
	xv,yv  = load_data_test()
	W      = load_w()
	zv     = ut.forward(xv,W)      		
	cm,Fsc = ut.metricas(yv,zv) 	
	save_measure(cm,Fsc)
		

if __name__ == '__main__':   
	 main()

