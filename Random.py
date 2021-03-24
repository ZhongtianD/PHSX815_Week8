

import numpy as np

from numpy.random import RandomState, SeedSequence, MT19937


    
class Gaussian:
    
    def __init__(self, seed = 1010, mu = 0, sigma = 1):
        self.random = RandomState(MT19937(SeedSequence(int(seed))))
        
        self.mu = mu
        self.sigma = sigma
    
    
    def loglike(self, sample):
         # log likelihood of a specific sample 
        return np.log(self.Gaussian_pdf(sample))
    
    def Gaussian_pdf(self, sample):
        #probability density of a sample
        a = sample
        mu = self.mu
        sigma = self.sigma
        
        return np.exp( - (a - mu)**2 / (2 * sigma**2))*1/(sigma*np.sqrt(2*np.pi))

    # function returns a random number from the Gaussian distr. using slice sampling method
    def Gaussian_sample(self, init, Nsample):
        samples = np.zeros(Nsample)
        
        pdf = self.Gaussian_pdf
        random = self.random
        sigma  = self.sigma

        # initialize
        x_0 = init
        
        
        #generate a sequence of Nsample samples.
        for i in range(Nsample):
            p_0 = pdf(x_0)
            
            
            #pick a random place on the vertical line
            p = p_0*random.rand()
            
            # set a  horizontal slice
            r = random.rand()
            x_1 = x_0 - r*sigma 
            x_2 = x_0+(1-r)*sigma
            
            #increase length of the slice
            p_1 = pdf(x_1)
            while p_1 > p:
                x_1 = x_1 - sigma
                p_1 = pdf(x_1)
            p_2 = pdf(x_2)
            while p_2 > p:
                x_2 = x_2 + sigma
                p_2 = pdf(x_2)
            
            #try a sample
            while True:
                x = random.rand()*(x_2-x_1)+x_1
                p_0 = pdf(x)
            # if x is actually in the slice, take it, else adjust the length of the slice and find a new one.
                if p_0 > p:
                    x_0 = x
                    break
                elif x>x_0:
                    x_2 = x
                elif x<x_0:
                    x_1 = x


            samples[i] = x_0

        return samples
    
class Gaussian2:
    
    def __init__(self, seed = 1010, sigma = 1, mu0 = 0, sigma0 = 0.1 ):
        self.random = RandomState(MT19937(SeedSequence(int(seed))))
        
        self.seed = seed
        self.mu0 = mu0
        self.sigma = sigma
        self.sigma0 = sigma0
        
        self.Gmu = Gaussian(mu = mu0, sigma = sigma0)
    
    
    def loglike(self, sample):
         # log likelihood of a specific sample 
        return np.log(self.pdf(sample))
    
    def pdf(self, sample):
        #probability density of a sample
        mu = sample[0]
        Gmu = self.Gmu
        sigma = self.sigma
        x = sample[1]
        px = np.exp( - (x - mu)**2 / (2 * sigma**2))*1/(sigma*np.sqrt(2*np.pi))
        
        return px*Gmu.Gaussian_pdf(mu)

    # function returns a random number from the Gaussian distr. using slice sampling method
    def Sample(self, init, Nsample):
        samples = np.zeros((2,Nsample))
        
        pdf = self.pdf
        random = self.random
        sigma = [self.sigma0,self.sigma]

        # initialize
        x_0 = init
        
        
        #generate a sequence of Nsample samples.
        for i in range(Nsample):
            
            
            p_0 = pdf(x_0)
            
            for j in range(2):
            
                #pick a random place on the vertical line
                p = p_0*random.rand()
            
                # set a  horizontal slice
                r = random.rand()
                x_1 = x_0.copy()
                x_1[j] = x_1[j] - r*sigma[j] 
                x_2 = x_0.copy()
                x_2[j] = x_2[j]+(1-r)*sigma[j]
            
                #increase length of the slice
                p_1 = pdf(x_1)
                while p_1 > p:
                    x_1[j] = x_1[j] - sigma[j]
                    p_1 = pdf(x_1)
                p_2 = pdf(x_2)
                while p_2 > p:
                    x_2[j] = x_2[j] + sigma[j]
                    p_2 = pdf(x_2)
            
                #try a sample
                x_3 = x_0.copy()
                while True:
                    x = random.rand()*(x_2[j]-x_1[j])+x_1[j]
                    x_3[j] = x
                    p_0 = pdf(x_3)
                # if x is actually in the slice, take it, else adjust the length of the slice and find a new one.
                    if p_0 > p:
                        x_0[j] = x
                        break
                    elif x>x_0[j]:
                        x_2[j] = x
                    elif x<x_0[j]:
                        x_1[j] = x


            samples[:,i] = x_0.copy()

        return samples[1,:]
