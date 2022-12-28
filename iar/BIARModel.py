import numpy as np
import random
from numpy.linalg import svd
import scipy
from scipy.optimize import minimize,minimize_scalar
from numpy.linalg import inv
import math
from numba import njit,jit

@njit                                                                                                                              
def BIARsample(n,sT,phi_R,phi_I,delta1=0,delta2=0,rho=0):
    delta=np.diff(sT)
    x=np.zeros(shape=(2,n))
    F=np.zeros(shape=(2,2))
    phi=complex(phi_R, phi_I)
    if abs(phi) > 1 :
         raise ValueError('Mod of Phi must be less than one')
    Phi=abs(phi)
    psi=np.arccos(phi_R/Phi)
    if phi_I<0:
        psi=-np.arccos(phi_R/Phi)
    e_R=np.random.normal(0, 1, n)
    e_I=np.random.normal(0, 1, n)
    state_error=np.vstack((e_R,e_I)).T
    Sigma=np.zeros(shape=(2,2))
    Sigma[0,0]=1
    Sigma[1,1]=1
    Sigma[0,1]=rho*np.sqrt(Sigma[0,0])*np.sqrt(Sigma[1,1])
    Sigma[1,0]=Sigma[0,1]
    B=np.linalg.svd(Sigma)
    A=np.zeros(shape=(2,2))
    np.fill_diagonal(A, np.sqrt(B[1]))
    Sigma_root=np.dot(B[0],np.dot(A,B[0].T))
    state_error=np.dot(Sigma_root,state_error.T)
    w_R=np.random.normal(0, 1, n)
    w_I=np.random.normal(0, 1, n)
    observation_error=np.vstack((w_R,w_I)).T
    Sigma0=np.zeros(shape=(2,2))
    Sigma0[0,0]=delta1**2
    Sigma0[1,1]=delta2**2
    B0=np.linalg.svd(Sigma0)
    A0=np.zeros(shape=(2,2))
    np.fill_diagonal(A0, np.sqrt(B0[1]))
    Sigma0_root=np.dot(B0[0],np.dot(A0,B0[0].T))
    observation_error=np.dot(Sigma0_root,observation_error.T)
    G=np.identity(2)
    y=np.zeros(shape=(2,n))
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
        y[0,i]=x[0,i]+observation_error[0,i]
        y[1,i]=x[1,i]+observation_error[1,i]
        #y[0:2,i]=np.dot(G,x[0:2,i])+observation_error[0:2,i]
    y[0,n-1]=x[0,n-1]+observation_error[0,n-1]
    y[1,n-1]=x[1,n-1]+observation_error[1,n-1]
    return y, sT,Sigma

@njit
def myCov(x,y):
    n = len(x)
    covarianza = 0
    mediaX = np.mean(x)
    mediaY = np.mean(y)
    for i in range(n):
        covarianza += (x[i]-mediaX)*(y[i]-mediaY)
    return covarianza/(n-1)

#Replace of np.cov(...) because it does'nt work in np python mode from numba
@njit
def mycov(x,y):
    lala1 = x
    lala2 = y
    my_cov = myCov(lala1,lala2)
    varianzaY0 = myCov(lala1,lala1)
    varianzaY1 = myCov(lala2,lala2)

    arrayCov = np.zeros(shape=(2,2))
    arrayCov[0,0] = varianzaY0
    arrayCov[1,1] = varianzaY1
    arrayCov[1,0] = arrayCov[0,1] = my_cov
    return arrayCov

@jit(forceobj=True)
def BIARphikalman(x,y1,y2,t,yerr1,yerr2,zero_mean=True,transform_par=False):
    sigmay=np.zeros(shape=(2,2))
    sigmay=mycov(y1,y2)
    if zero_mean == False:
        y1=y1-np.mean(y1)
        y2=y2-np.mean(y2)
    if transform_par==True:
        x[0]=np.tanh(x[0])
        x[1]=np.tanh(x[1])
    n=len(y1)
    Sighat=np.dot(sigmay,np.identity(2))
    xhat=np.zeros(shape=(2,n))
    delta=np.diff(t)
    Q=Sighat
    phi_R=x[0]
    phi_I=x[1]
    F=np.zeros(shape=(2,2))
    G=np.identity(2)
    phi=complex(phi_R, phi_I)
    Phi=np.sqrt(phi_R**2+phi_I**2)
    if phi_I>=0:
        psi=np.arccos(phi_R/Phi)
    if phi_I<0:
        psi=-np.arccos(phi_R/Phi)
    sum_Lambda=0
    sum_error=0
    if np.isnan(phi) == True:
        phi=1.1
    y=np.vstack((y1,y2))
    innov = np.zeros(2)
    if abs(phi) < 1:
        for i in range(n-1):
            derr=np.zeros(shape=(2,2))
            derr[0,0]=yerr1[i+1]**2
            derr[1,1]=yerr2[i+1]**2
            Lambda=np.dot(np.dot(G,Sighat),G.transpose())+derr
            if (np.linalg.det(Lambda) <= 0) or ((np.isnan(Lambda) == True).any() == True):
                sum_Lambda=n*1e10
                break
            phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
            phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
            phi2=1-abs(phi**delta[i])**2
            F[0,0]=phi2_R
            F[0,1]=-phi2_I
            F[1,0]=phi2_I
            F[1,1]=phi2_R
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(np.linalg.det(Lambda))
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            innov[0] = y[0,i]-xhat[0,i]
            innov[1] = y[1,i]-xhat[1,i]
            sum_error= sum_error + np.dot(np.dot(innov.T,inv(Lambda)),innov)
            temp1 = F[0,0]*xhat[0,i]+F[0,1]*xhat[1,i]
            temp2 = F[1,0]*xhat[0,i]+F[1,1]*xhat[1,i]
            temp3 = np.dot(np.dot(Theta,inv(Lambda)),innov)
            xhat[0,i+1]=temp1+temp3[0]
            xhat[1,i+1]=temp2+temp3[1]
            B=np.dot(Sighat,inv(Lambda))
            if np.sum(derr)==0:
                B=np.identity(2)
            A=Sighat-np.dot(B,Sighat.transpose())
            Sighat=Qt + np.dot(np.dot(F,A),F.transpose())
        out=(sum_Lambda + sum_error)/n
        if np.isnan(sum_Lambda) == True:
            out=1e10
    else:
        out=1e10
    return out

def BIARkalman(y1,y2,sT,delta1=0,delta2=0,zero_mean=True,niter=10,seed=1234):
    random.seed(seed)
    aux=1e10
    value=1e10
    br=0
    if np.sum(delta1)==0:
        delta1=np.zeros(len(y1))
    if np.sum(delta2)==0:
        delta2=np.zeros(len(y2))
    for i in range(niter):
        phi_R=2*np.random.uniform(0,1,1)-1
        phi_I=2*np.random.uniform(0,1,1)-1
        bnds = ((-0.9999, 0.9999), (-0.9999, 0.9999))
        out=minimize(BIARphikalman,np.array([phi_R, phi_I]),args=(y1,y2,sT,delta1,delta2,zero_mean),bounds=bnds,method='L-BFGS-B')
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

#Fit a BIAR model to a bivariate irregularly observed time series.
@jit(forceobj=True)
def BIARfit(x,y1,y2,t,yerr1,yerr2,zero_mean=True,transform_par=False):
    sigmay=np.zeros(shape=(2,2))
    sigmay=mycov(y1,y2)
    if zero_mean == False:
        y1=y1-np.mean(y1)
        y2=y2-np.mean(y2)
    if transform_par==True:
        x[0]=np.tanh(x[0])
        x[1]=np.tanh(x[1])
    n=len(y1)
    Sighat=np.dot(sigmay,np.identity(2))
    xhat=np.zeros(shape=(2,n))
    delta=np.diff(t)
    Q=Sighat
    phi_R=x[0]
    phi_I=x[1]
    F=np.zeros(shape=(2,2))
    G=np.identity(2)
    phi=complex(phi_R, phi_I)
    Phi=np.sqrt(phi_R**2+phi_I**2)
    if phi_I>=0:
        psi=np.arccos(phi_R/Phi)
    if phi_I<0:
        psi=-np.arccos(phi_R/Phi)
    sum_Lambda=0
    sum_error=0
    if np.isnan(phi) == True:
        phi=1.1
    y=np.vstack((y1,y2))
    innov = np.zeros(2)
    for i in range(n-1):
            derr=np.zeros(shape=(2,2))
            derr[0,0]=yerr1[i+1]**2
            derr[1,1]=yerr2[i+1]**2
            Lambda=np.dot(np.dot(G,Sighat),G.transpose())+derr
            if (np.linalg.det(Lambda) <= 0) or ((np.isnan(Lambda) == True).any() == True):
                sum_Lambda=n*1e10
                break
            phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
            phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
            phi2=1-abs(phi**delta[i])**2
            F[0,0]=phi2_R
            F[0,1]=-phi2_I
            F[1,0]=phi2_I
            F[1,1]=phi2_R
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(np.linalg.det(Lambda))
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            innov[0] = y[0,i]-xhat[0,i]
            innov[1] = y[1,i]-xhat[1,i]
            sum_error= sum_error + np.dot(np.dot(innov.T,inv(Lambda)),innov)
            temp1 = F[0,0]*xhat[0,i]+F[0,1]*xhat[1,i]
            temp2 = F[1,0]*xhat[0,i]+F[1,1]*xhat[1,i]
            temp3 = np.dot(np.dot(Theta,inv(Lambda)),innov)
            xhat[0,i+1]=temp1+temp3[0]
            xhat[1,i+1]=temp2+temp3[1]
            B=np.dot(Sighat,inv(Lambda))
            if np.sum(derr)==0:
                B=np.identity(2)
            A=Sighat-np.dot(B,Sighat.transpose())
            Sighat=Qt + np.dot(np.dot(F,A),F.transpose())
    yhat=np.dot(G,xhat)
    out=(sum_Lambda + sum_error)/n
    innov=y-np.dot(G,xhat)
    Innovcov=np.cov(innov)
    Innovcor=np.corrcoef(innov)[0,1]
    return Innovcor,Innovcov,yhat,xhat,Sighat,Theta,Lambda,Qt

# Funcion nueva
def BIARforecast(x,y1,y2,t,tahead):
    phi_R=x[0]
    phi_I=x[1]
    phi=complex(phi_R, phi_I)
    Phi=abs(phi)
    cor,cov,yhat,xhat,Sighat,Theta,Lambda,Qt = BIARfit(x=np.array([phi_R, phi_I]),y1=y1,y2=y2,t=t,yerr1=np.zeros(len(y1)),yerr2=np.zeros(len(y1)))
    n=np.shape(yhat)[1]
    G=np.identity(2)
    if phi_I>=0:
        psi=np.arccos(phi_R/Phi)
    if phi_I<0:
        psi=-np.arccos(phi_R/Phi)
    if(len(np.shape(tahead))==0):
        n1=1
        tahead=[tahead]
    else:
        n1=len(tahead)
    yhat1=np.zeros(shape=(2,n1))
    xhat1=np.zeros(shape=(2,n1))
    Lambda2=np.zeros(n1)
    y=np.vstack((y1,y2))
    n=len(y1)
    for i in range(n1):
        phi2_R=(Phi**tahead[i])*np.cos(tahead[i]*psi)
        phi2_I=(Phi**tahead[i])*np.sin(tahead[i]*psi)
        F=np.zeros(shape=(2,2))
        F[0,0]=phi2_R
        F[0,1]=-phi2_I
        F[1,0]=phi2_I
        F[1,1]=phi2_R
        xhat1[0:2,i]=np.dot(F,xhat[0:2,n-1])+np.dot(np.dot(Theta,inv(Lambda)),(y1[n-1]-np.dot(G,xhat[0:2,n-1])))
        yhat1[0:2,i]=np.dot(G,xhat1[0:2,i])
        Sighat2=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        Lambda2=np.dot(np.dot(G,Sighat2),(G).transpose())
    return yhat,yhat1,Lambda2,Sighat2

def BIARLL(yest,x,y1,y2,t,yerr1,yerr2,zero_mean=True,standardized=True):
    maskedarr = np.ma.array((y1,y2), mask=np.isnan((y1,y2)))
    sigmay=np.zeros(shape=(2,2))
    sigmay=np.ma.cov(maskedarr,allow_masked=True)
    if zero_mean == False:
        y1=y1-np.mean(y1)
        y2=y2-np.mean(y2)
    n=len(y1)
    Sighat=np.dot(sigmay,np.identity(2))
    xhat=np.zeros(shape=(2,n))
    delta=np.diff(t)
    Q=Sighat
    phi_R=x[0]
    phi_I=x[1]
    F=np.zeros(shape=(2,2))
    G=np.identity(2)
    phi=complex(phi_R, phi_I)
    Phi=np.sqrt(phi_R**2+phi_I**2)
    psi=np.arccos(phi_R/Phi)
    if phi_I<0:
        psi=-np.arccos(phi_R/Phi)
    if np.isnan(phi) == True:
        phi=1.1
    y=np.vstack((y1,y2))
    cte=-0.5*np.log(np.linalg.det(Sighat))
    sumll=cte
    if abs(phi) < 1:
        for i in range(n-1):
            yaux=y[0:2,i+1]
            derr=np.zeros(shape=(2,2))
            derr[0,0]=yerr1[i+1]**2
            derr[1,1]=yerr2[i+1]**2
            phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
            phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
            phi2=1-abs(phi**delta[i])**2
            F[0,0]=phi2_R
            F[0,1]=-phi2_I
            F[1,0]=phi2_I
            F[1,1]=phi2_R
            Qt=phi2*Q
            V=np.random.multivariate_normal((0,0), derr, 1)
            if (np.isnan(yaux) == True).all() == True:
                xhat[0:2,i+1]=yest-V[0]
                innov=yest-np.dot(G,xhat[0:2,i+1])
            if np.logical_and((np.isnan(yaux) == True)[0],(np.isnan(yaux) == False)[1]) == True:
                xhat[0,i+1]=yest-V[0][0]
                xhat[1,i+1]=yaux[1]-V[0][1]
                innov=(yest,yaux[1])-np.dot(G,xhat[0:2,i+1])
            if np.logical_and((np.isnan(yaux) == False)[0],(np.isnan(yaux) == True)[1]) == True:
                xhat[0,i+1]=yaux[0]-V[0][0]
                xhat[1,i+1]=yest-V[0][1]
                innov=(yaux[0],yest)-np.dot(G,xhat[0:2,i+1])
            if (np.isnan(yaux) == True).any() == False:
                xhat[0:2,i+1]=yaux-V[0]
                innov=yaux-np.dot(G,xhat[0:2,i+1])
            innov_tr=xhat[0:2,i+1]-np.dot(F,xhat[0:2,i])
            if np.linalg.det(derr) != 0:
                cte=-0.5*n*np.log(np.linalg.det(Qt))-0.5*n*np.log(np.linalg.det(derr))
                comp2=-0.5*np.dot(innov_tr.T,np.dot(inv(Qt),innov_tr))-0.5*np.dot(innov,np.dot(inv(derr),innov))
            else:
                cte=-0.5*n*np.log(np.linalg.det(Qt))
                comp2=-0.5*np.dot(innov_tr.T,np.dot(inv(Qt),innov_tr))
            sumll=sumll+cte+comp2
        out=-sumll
    else:
        out=1e10
    return out

#This function estimates the missing values of a BIAR process given specific values of phi.R and phi.I
def BIARinterpolation(x,y1,y2,t,yerr1=0,yerr2=0,zero_mean=True,standardized=True,seed=1234,niter=10,yini1=0,yini2=0,nsmooth=1):
    random.seed(seed)
    aux=1e10
    value=1e10
    br=0
    if np.sum(yerr1)==0:
        yerr1=np.zeros(len(y1))
    if np.sum(yerr2)==0:
        yerr2=np.zeros(len(y2))
    if nsmooth==2:
        if np.logical_or(np.isnan(y1[0]),np.isnan(y2[0])):
            par=(np.nanmean(y1),np.nanmean(y2))
        else:
            for i in range(niter):
                if yini1==0:
                    yini1=np.random.normal(0, 1, 1)
                if yini2==0:
                    yini2=np.random.normal(0, 1, 1)
                bnds = ((-np.Inf, np.Inf), (-np.Inf, np.Inf))
                out=minimize(BIAR_LL,np.array([yini1, yini2]),args=(x,y1,y2,t,yerr1,yerr2,zero_mean,standardized),bounds=bnds,method='L-BFGS-B')
                value=out.fun
                if aux > value:
                    par=out.x
                    aux=value
                    br=br+1
                if aux <= value and br>1 and i>math.trunc(niter/2):
                    break
    if nsmooth==1:
        if np.isnan(y1[0]):
            par=np.nanmean(y1)
        if np.isnan(y2[0]):
            par=np.nanmean(y2)
        if np.logical_and(~np.isnan(y1[0]),~np.isnan(y2[0])):
            out=minimize_scalar(BIARLL,args=(x,y1,y2,t,yerr1,yerr2,zero_mean,standardized),method="Brent",tol=0.0001220703)
            par=out.x
            aux=out.fun
    if aux == 1e10:
        par=np.zeros(nsmooth)
    return par,aux
