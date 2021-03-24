# imports of external packages to use in our code
import sys
import numpy as np
import matplotlib.pyplot as plt

# import Gaussian class 
sys.path.append(".")
from Random import Gaussian

# main function 
if __name__ == "__main__":
    
    #array of true mu values
    mu_true = np.arange(-10,10,0.1)
    Y = []
    Sample = []
    for i in mu_true:
        #build a gaussian class for each mu
        G = Gaussian( mu=i,sigma = 2, seed  = i*10+1000)
        #take 100000 measurements
        Sample.append(G.Gaussian_sample(i,100000))
        Y.append(np.full(100000,i))
    
    Y = np.array(Y)
    Y = Y.reshape(20000000,-1)
    Sample = np.array(Sample)
    Sample = Sample.reshape(20000000,-1)
    

    # make figure
    plt.figure()
    plt.hist2d(Y[:, 0],Sample[:, 0], bins=100)

    plt.xlabel('True value of $\\mu$')
    plt.ylabel('Sampleed of $\\mu$')
    plt.title('Neyman_construction of Gaussian')

    plt.show()
    plt.savefig('Neyman.png')
    
