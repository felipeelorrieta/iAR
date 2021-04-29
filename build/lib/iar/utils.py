import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
from scipy.stats import gaussian_kde
from numpy.linalg import inv

def gentime(n,lambda1=130,lambda2=6.5,w1=0.15,w2=0.85):
    aux1=np.random.exponential(scale=lambda1,size=n)
    aux2=np.random.exponential(scale=lambda2,size=n)
    aux = np.hstack((aux1,aux2))
    prob = np.hstack((np.repeat(w1, n),np.repeat(w2, n)))/n
    dT=np.random.choice(aux,n,p=prob)
    sT=np.cumsum(dT)
    return sT

def harmonicfit(t,m,f):
    ws = pd.DataFrame({
    'x': m,
    't': t})
    ols_fit=sm.ols('x ~ t', data=ws).fit()
    m = ols_fit.resid
    ws = pd.DataFrame({
    'x': m,
    'y1': np.sin(2*np.pi*t*f),
    'y2': np.cos(2*np.pi*t*f),
    'y3': np.sin(4*np.pi*t*f),
    'y4': np.cos(4*np.pi*t*f),
    'y5': np.sin(6*np.pi*t*f),
    'y6': np.cos(6*np.pi*t*f),
    'y7': np.sin(8*np.pi*t*f),
    'y8': np.cos(8*np.pi*t*f)
    })
    ols_fit=sm.ols('x ~ y1+y2+y3+y4+y5+y6+y7+y8', data=ws).fit()
    res = ols_fit.resid
    return res,t
