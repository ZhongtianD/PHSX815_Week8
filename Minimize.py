# imports of external packages to use in our code
from scipy.optimize import minimize
#from iminuit import Minuit


# main function 
if __name__ == "__main__":
    
    def fcn(x):
        return  (2 - x[0])**2+105*(x[1]-x[0]**2)**2
    
    #It minimizes the  function 
    res = minimize(fcn,(0,0), tol=1e-6)
    
    print(res.x)
    
    
    #Using Minuit
    #fcn.errordef = 1
    #m = Minuit(fcn, x=0,y=0)
    #m.migrad()  # run optimiser
    #print(m.values)
    
