import numpy as np
import random
import math
from numpy.linalg import svd
import scipy
from scipy.optimize import minimize,minimize_scalar
from numpy.linalg import inv
from numba import njit,jit

@njit
def CIARsample(n,sT,phi_R,phi_I,rho=0,c=0.5):
    delta=np.diff(sT)
    x=np.zeros(shape=(2,n))
    F=np.zeros(shape=(2,2))
    phi=complex(phi_R, phi_I)
    if abs(phi) > 1 :
         raise ValueError('Mod of Phi must be less than one')
    Phi=abs(phi)
    psi=np.arccos(phi_R/Phi)
    e_R=np.random.normal(0, 1, n)
    e_I=np.random.normal(0, 1, n)
    state_error=np.vstack((e_R,e_I)).T
    Sigma=np.zeros(shape=(2,2))
    Sigma[0,0]=1
    Sigma[1,1]=c
    Sigma[0,1]=rho*np.sqrt(Sigma[0,0])*np.sqrt(Sigma[1,1])
    Sigma[1,0]=Sigma[0,1]
    B=np.linalg.svd(Sigma)
    A=np.zeros(shape=(2,2))
    np.fill_diagonal(A, np.sqrt(B[1]))
    Sigma_root=np.dot(B[0],np.dot(A,B[0].T))
    state_error=np.dot(Sigma_root,state_error.T)
    G=np.zeros(shape=(1,2))
    G[0,0]=1
    y=np.zeros(n)
    x[0:2,0]=state_error[0:2,0]
    for i in range(n-1):
        phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
        phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
        phi2=1-abs(phi**delta[i])**2
        F[0,0]=phi2_R
        F[0,1]=-phi2_I
        F[1,0]=phi2_I
        F[1,1]=phi2_R
        temp1 = F[0,0]*x[0,i]+F[0,1]*x[1,i]
        temp2 = F[1,0]*x[0,i]+F[1,1]*x[1,i]
        x[0,i+1]=temp1+np.sqrt(phi2)*state_error[0,i]
        x[1,i+1]=temp2+np.sqrt(phi2)*state_error[1,i]
        y[i] = x[0,i]
    y[n-1]=x[0,n-1]
    return y, sT,Sigma

@jit(forceobj=True)
def CIARphikalman(x,y,t,yerr,zero_mean=True,standardized=True,c=0.5):
    n=len(y)
    Sighat=np.zeros(shape=(2,2))
    Sighat[0,0]=1
    Sighat[1,1]=c
    if standardized == False:
         Sighat=np.var(y)*Sighat
    if zero_mean == False:
         y=y-np.mean(y)
    xhat=np.zeros(shape=(2,n))
    delta=np.diff(t)
    Q=Sighat
    phi_R=x[0]
    phi_I=x[1]
    F=np.zeros(shape=(2,2))
    G=np.zeros(shape=(1,2))
    G[0,0]=1
    phi=complex(phi_R, phi_I)
    Phi=abs(phi)
    psi=np.arccos(phi_R/Phi)
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
            temp = 0.1
            phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
            phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
            phi2=1-abs(phi**delta[i])**2
            F[0,0]=phi2_R
            F[0,1]=-phi2_I
            F[1,0]=phi2_I
            F[1,1]=phi2_R
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(Lambda[0][0])
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            sum_error= sum_error + ((y[i]-np.dot(G,xhat[0:2,i]))**2/(Lambda[0][0]))[0]
            xhat[0:2,i+1]=np.dot(F,xhat[0:2,i])+np.dot(np.dot(Theta,inv(Lambda)),(y[i]-np.dot(G,xhat[0:2,i])))
            Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        yhat=np.dot(G,xhat)
        out=(sum_Lambda + sum_error)/n
        out=out
        if np.isnan(sum_Lambda) == True:
            out=1e10
    else:
        out=1e10
    return out

def CIARkalman(y,sT,yerr=np.array([0.0]),zero_mean=True,standardized=True,c=0.5,niter=10,seed=1234):
    random.seed(seed)
    aux=1e10
    value=1e10
    br=0
    if np.sum(yerr)==0:
        yerr=np.zeros(len(y))
    for i in range(niter):
        phi_R=2*np.random.uniform(0,1,1)-1
        phi_I=2*np.random.uniform(0,1,1)-1
        bnds = ((-0.9999, 0.9999), (-0.9999, 0.9999))
        out=minimize(CIARphikalman,np.array([phi_R, phi_I]),args=(y,sT,yerr,zero_mean,standardized,c),bounds=bnds,method='L-BFGS-B')
        value=out.fun
        if aux > value:
            par=out.x
            aux=value
            br=br+1
        if aux <= value and br>1 and i>math.trunc(niter/2):
            break                                                                                                                                                                                          
    if aux == 1e10:
        par=np.zeros(2)
    return par[0],par[1],aux

@jit(forceobj=True)
def CIARfit(x,y,t,yerr=np.array([0.0]),standardized=True,c=1):
    n=len(y)
    Sighat=np.zeros(shape=(2,2))
    Sighat[0,0]=1
    Sighat[1,1]=c
    if standardized == False:
         Sighat=np.var(y)*Sighat
    if np.sum(yerr)==0:
        yerr=np.zeros(len(y))
    xhat=np.zeros(shape=(2,n))
    delta=np.diff(t)
    Q=Sighat
    phi_R=x[0]
    phi_I=x[1]
    F=np.zeros(shape=(2,2))
    G=np.zeros(shape=(1,2))
    G[0,0]=1
    phi=complex(phi_R, phi_I)
    Phi=abs(phi)
    psi=np.arccos(phi_R/Phi)
    if np.isnan(phi) == True:
        phi=1.1
    if abs(phi) < 1:
        for i in range(n-1):
            Lambda=np.dot(np.dot(G,Sighat),G.transpose())+yerr[i+1]**2
            if (Lambda[0][0] <= 0) or (np.isnan(Lambda) == True):
                break
            phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
            phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
            phi2=1-abs(phi**delta[i])**2
            F[0,0]=phi2_R
            F[0,1]=-phi2_I
            F[1,0]=phi2_I
            F[1,1]=phi2_R
            Qt=phi2*Q
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            xhat[0:2,i+1]=np.dot(F,xhat[0:2,i])+np.dot(np.dot(Theta,inv(Lambda)),(y[i]-np.dot(G,xhat[0:2,i])))
            Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        yhat=np.dot(G,xhat)
    else:
        raise ValueError('Mod of Phi must be less than one')
    return yhat,Sighat,xhat,Theta,Lambda,Qt

@jit(forceobj=True)
def CIARforecast(phi_R,phi_I,y,t,tahead=1):
    yhat,Sighat,xhat,Theta,Lambda,Qt = CIARfit(np.array([phi_R, phi_I]),y,t)
    n=np.shape(yhat)[1]
    F=np.zeros(shape=(2,2))
    G=np.zeros(shape=(1,2))
    G[0,0]=1
    phi=complex(phi_R, phi_I)
    Phi=abs(phi)
    psi=np.arccos(phi_R/Phi)
    if np.isnan(phi) == True:
        phi=1.1
    delta=tahead
    phi2_R=(Phi**delta)*np.cos(delta*psi)
    phi2_I=(Phi**delta)*np.sin(delta*psi)
    n1=len(tahead)
    yhat1=np.zeros(n1)
    xhat1=np.zeros(shape=(2,n1))
    Lambda2=np.zeros(n1)
    for i in range(n1):
        F[0,0]=phi2_R[i]
        F[0,1]=-phi2_I[i]
        F[1,0]=phi2_I[i]
        F[1,1]=phi2_R[i]
        xhat1[0:2,i]=np.dot(F,xhat[0:2,n-1])+np.dot(np.dot(Theta,inv(Lambda)),(y[n-1]-np.dot(G,xhat[0:2,n-1])))
        yhat1[i]=np.dot(G[0],xhat1[0:2,i])
        Sighat2=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        Lambda2[i]=np.dot(np.dot(G[0],Sighat2),(G[0]).transpose())
    return yhat,yhat1,Lambda2,Sighat2

@jit(forceobj=True)
def CIARphikalman2(yest,x,y,t,yerr,zero_mean=True,standardized=True,c=0.5):
    n=len(y)
    Sighat=np.zeros(shape=(2,2))
    Sighat[0,0]=1
    Sighat[1,1]=c
    y_copy = y[~np.isnan(y)]
    if standardized == False:
         Sighat=np.var(y_copy)*Sighat
    if zero_mean == False:
         y=y-np.mean(y_copy)
    xhat=np.zeros(shape=(2,n))
    delta=np.diff(t)
    Q=Sighat
    phi_R=x[0]
    phi_I=x[1]
    F=np.zeros(shape=(2,2))
    G=np.zeros(shape=(1,2))
    G[0,0]=1
    phi=complex(phi_R, phi_I)
    Phi=abs(phi)
    psi=np.arccos(phi_R/Phi)
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
            temp = 0.1
            phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
            phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
            phi2=1-abs(phi**delta[i])**2
            F[0,0]=phi2_R
            F[0,1]=-phi2_I
            F[1,0]=phi2_I
            F[1,1]=phi2_R
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(Lambda[0][0])
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            yaux = y[i]
            innov = yaux - np.dot(G,xhat[0:2,i])
            if np.isnan(yaux) == True:
                innov = yest - np.dot(G,xhat[0:2,i])
            sum_error= sum_error + ((innov)**2/(Lambda[0][0]))[0]
            xhat[0:2,i+1]=np.dot(F,xhat[0:2,i])+np.dot(np.dot(Theta,inv(Lambda)),(innov))
            Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        yhat=np.dot(G,xhat)
        out=(sum_Lambda + sum_error)/n
        out=out
        if np.isnan(sum_Lambda) == True:
            out=1e10
    else:
        out=1e10
    return out

def CIARinterpolation(x, y, t, delta= 0, yini=0 , zero_mean=True, standardized=True, c=1, seed=1234):
    random.seed(seed)
    aux = 1e+10
    value = 1e+10
    br = 0
    if np.sum(delta) == 0:
        delta = np.zeros(len(y))
    if yini==0:
        yini = np.random.normal(0, 1, 1)
    bnds = (-np.Inf, np.Inf)
    out = minimize(CIARphikalman2, yini, args=(x, y, t, delta, zero_mean, standardized, c), bounds=bnds, method='BFGS')
    par = out.x
    aux= out.fun
    return par, aux
