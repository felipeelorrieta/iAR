import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from numpy.linalg import inv

def IAR_sample(phi,n,sT):
    Sigma=np.zeros(shape=(n,n))
    for i in range(np.shape(Sigma)[0]):
        d=sT[i]-sT[i:n]
        Sigma[i,i:n]=phi**abs(d)
        Sigma[i:n,i]=Sigma[i,i:n]
    b,v=LA.eig(Sigma)
    A=np.dot(np.dot(v,np.diag(np.sqrt(b))),v.transpose())
    e=np.random.normal(0, 1, n)
    y=np.dot(A,e)
    return y, sT

def IAR_phi_loglik(x,y,sT,delta,include_mean=False,standarized=True):
    n=len(y)
    sigma=1
    mu=0
    if standarized == False:
        sigma=np.var(y,ddof=1)
    if include_mean == True:
        mu=np.mean(y)
    d=np.diff(sT)
    delta=delta[1:n]
    phi=x**d
    yhat=mu+phi*(y[0:(n-1)]-mu)
    y2=np.vstack((y[1:n],yhat))
    cte=0.5*n*np.log(2*np.pi)
    s1=cte+0.5*np.sum(np.log(sigma*(1-phi**2)+delta**2)+(y2[0,]-y2[1,])**2/(sigma*(1-phi**2)+delta**2))
    return s1

def IAR_loglik(y,sT,delta,include_mean=False,standarized=True):
    if np.sum(delta)==0:
        delta=np.zeros(len(y))
    out=minimize_scalar(IAR_phi_loglik,args=(y,sT,delta,include_mean,standarized),bounds=(0,1),method="bounded",options={"xatol":0.0001220703})
    return out.x

def IAR_phi_kalman(x,y,yerr,t,zero_mean=True,standarized=True,c=0.5):
    n=len(y)
    Sighat=np.zeros(shape=(1,1))
    Sighat[0,0]=1
    if standarized == False:
         Sighat=np.var(y)*Sighat
    if zero_mean == False:
         y=y-np.mean(y)
    xhat=np.zeros(shape=(1,n))
    delta=np.diff(t)
    Q=Sighat
    phi=x
    F=np.zeros(shape=(1,1))
    G=np.zeros(shape=(1,1))
    G[0,0]=1
    sum_Lambda=0
    sum_error=0
    if np.isnan(phi) == True:
        phi=1.1
    if abs(phi) < 1:
        for i in range(n-1):
            Lambda=np.dot(np.dot(G,Sighat),G.transpose())+yerr[i+1]**2 
            if (Lambda <= 0) or (np.isnan(Lambda) == True):
                sum_Lambda=n*1e10
                break
            phi2=phi**delta[i]
            F[0,0]=phi2
            phi2=1-phi**(delta[i]*2)
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(Lambda)
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            sum_error= sum_error + (y[i]-np.dot(G,xhat[0:1,i]))**2/Lambda
            xhat[0:1,i+1]=np.dot(F,xhat[0:1,i])+np.dot(np.dot(Theta,inv(Lambda)),(y[i]-np.dot(G,xhat[0:1,i])))
            Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        yhat=np.dot(G,xhat)
        out=(sum_Lambda + sum_error)/n
        if np.isnan(sum_Lambda) == True:
            out=1e10
    else:
        out=1e10
    return out


def IAR_kalman(y,sT,delta=0,zero_mean=True,standarized=True):
    if np.sum(delta)==0:
        delta=np.zeros(len(y))
    out=minimize_scalar(IAR_phi_kalman,args=(y,delta,sT,zero_mean,standarized),bounds=(0,1),method="bounded",tol=0.0001220703)
    return out.x


