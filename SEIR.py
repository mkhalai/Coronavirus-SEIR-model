import numpy as np
from scipy import optimize
import pandas as pd
from scipy import integrate

class Covid19:
    """
    Find optimal parameters for infection rate (beta),
    incubation (sigma) and recovery (gamma) for any given province in China.

    All data sourced from CSSE John Hopkins University
    (https://github.com/CSSEGISandData)
    """
    def __init__(self,frame,initial_params,N=100000):
        """
            frame: pd.DataFrame with columns cases,recovered
            N: population size def: 100k, (sufficiently large)
            initial_params : [beta,gamma,sigma] (initial guess) as list.

        """
        self.data = frame
        self.infections = self.data.iloc[:,0].values
        self.I0 = self.infections[0]
        self.N = N
        self.params = initial_params

    def derivative(self,t,Y,N,beta,gamma,sigma):
        S,E,I,R=Y[0],Y[1],Y[2],Y[3]
        dsdt = -beta * S * I / N
        dedt = beta * S * I / N - sigma * E
        didt = sigma*E - gamma*I
        drdt = gamma * I
        return [dsdt,dedt,didt,drdt]

    def fit_cumulated_infected(self,x,args):
        """
        y0 = [S,E,I,R]
        args = [N,I0,maxt]
        """
        N,I0,maxt = args[0],args[1],args[2]
        params = [N,x[0],x[1],x[2]]
        span = np.linspace(1,maxt,maxt)
        res = integrate.solve_ivp(
                lambda t, Y: self.derivative(t,Y,*params),
                t_span = [1,maxt],
                y0 = [N-I0,0,I0,0],
                t_eval = span

            )
        cumulative_infected = res.y[2]+res.y[3]
        return cumulative_infected

    def loss_function(self,x,N,I0,maxt):
        """
        Objective function.
        Computes fitted values for beta,gamma, sigma.
        Returns loss as RMSE

        x := [beta,gamma,sigma]
        args := [N,I0,maxt]

        """
        args = [N,I0,maxt]
        beta,gamma,sigma = x
        fitted = self.fit_cumulated_infected(x,args)

        loss = np.sqrt(np.mean((fitted-self.infections)**2))
        return loss

    def optimize(self):
        """
        optimize parameters [beta,gamma,sigma]
        """
        I0 = self.data.iloc[0,0]
        dates = self.data.index
        maxt = dates.size
        infections = self.data.iloc[:,0].values
        args = [self.N,I0,maxt]

        result = optimize.minimize(lambda x: self.loss_function(x,*args),
                                   x0 = self.params,
                                   method = 'Nelder-Mead')
        return result.x

    def get_curves(self,bgs):
        #bgs = [beta, gamma, sigma]
        params = [self.N,bgs[0],bgs[1],bgs[2]]
        maxt = self.infections.size
        span = np.linspace(1,maxt,maxt)
        res = integrate.solve_ivp(
                lambda t, Y: self.derivative(t,Y,*params),
                t_span = [1,maxt],
                y0 = [self.N-self.I0,0,self.I0,0],
                t_eval = span
            )
        return res.y[2]+res.y[3]
