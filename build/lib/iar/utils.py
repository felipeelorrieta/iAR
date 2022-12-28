import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
from scipy.stats import gaussian_kde
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import statsmodels.formula.api as sm

def gentime(n,distribution="expmixture",lambda1=130,lambda2=6.5,w1=0.15,w2=0.85,a=0,b=1):
    if distribution=="expmixture":
        aux1=np.random.exponential(scale=lambda1,size=n)
        aux2=np.random.exponential(scale=lambda2,size=n)
        aux = np.hstack((aux1,aux2))
        prob = np.hstack((np.repeat(w1, n),np.repeat(w2, n)))/n
        dT=np.random.choice(aux,n,p=prob)
        sT=np.cumsum(dT)
    if distribution=="uniform":
        sT=np.cumsum(np.random.uniform(low=a,high=b,size=n))
    if distribution=="exponential":
        sT=np.cumsum(np.random.exponential(scale=lambda1,size=n))
    if distribution=="gamma":
        sT=np.cumsum(np.random.gamma(shape = a, scale = b,size=n))
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

def foldlc(t,m,merr,f1,plot=True,nameP='folded.pdf'):
    P=1/f1
    fold=((t-t[0])%P)/P
    fold1=fold+1
    fold2=np.hstack((fold.tolist(),fold1.tolist()))
    m2=np.hstack((m.tolist(),m.tolist()))
    merr2=np.hstack((merr.tolist(),merr.tolist()))
    pos=np.argsort(fold2)
    fold2=fold2[pos]
    m2=m2[pos]
    merr2=merr2[pos]
    data = pd.DataFrame({
    't': fold2,
    'm': m2,'merr': merr2})
    if plot==True:
       pdf = matplotlib.backends.backend_pdf.PdfPages(nameP) 
       fig = plt.figure()
       plt.plot(fold2,m2,"o-")
       pdf.savefig(1)
       pdf.close()
    return data

def pairingits(lc1,lc2,tol=0.1):
    t1=lc1[0]
    t2=lc2[0]
    A1=np.vstack((t1,np.ones(len(t1)),np.arange(len(t1)))).T
    A2=np.vstack((t2,np.ones(len(t2))+1,np.arange(len(t2)))).T
    A=np.vstack((A1,A2))
    A=A[A[:,0].argsort()]
    fin=np.array([], dtype=np.int64).reshape(0,6)
    i=1
    while i<np.shape(A)[0]:
        if A[i-1,1]!=A[i,1]:
            dt=np.diff((A[i-1,0],A[i,0]))[0]
            if abs(dt)<tol:
                if A[i,1]>A[i-1,1]:
                    par=(lc1[0][int(A[i-1,2])],lc1[1][int(A[i-1,2])],lc1[2][int(A[i-1,2])],lc2[0][int(A[i,2])],lc2[1][int(A[i,2])],lc2[2][int(A[i,2])])
                if A[i,1]<A[i-1,1]:
                    par=(lc1[0][int(A[i,2])],lc1[1][int(A[i,2])],lc1[2][int(A[i,2])],lc2[0][int(A[i-1,2])],lc2[1][int(A[i-1,2])],lc2[2][int(A[i-1,2])])
                i=i+1
            else:
                if A[i-1,1]==1:
                    par=(lc1[0][int(A[i-1,2])],lc1[1][int(A[i-1,2])],lc1[2][int(A[i-1,2])],np.NaN,np.NaN,np.NaN)
                if A[i-1,1]==2:
                    par=(np.NaN,np.NaN,np.NaN,lc2[0][int(A[i-1,2])],lc2[1][int(A[i-1,2])],lc2[2][int(A[i-1,2])])
        else:
            if np.logical_and(A[i-1,1]==1,A[i,1]==1):
                par=(lc1[0][int(A[i-1,2])],lc1[1][int(A[i-1,2])],lc1[2][int(A[i-1,2])],np.NaN,np.NaN,np.NaN)
            if A[i-1,1]==2:
                par=(np.NaN,np.NaN,np.NaN,lc2[0][int(A[i-1,2])],lc2[1][int(A[i-1,2])],lc2[2][int(A[i-1,2])])
        fin=np.vstack((fin,par))
        i=i+1
    i=i-1
    if i==(np.shape(A)[0]-1):
        if np.logical_or(A[i-1,1]==A[i,1],np.logical_and(A[i-1,1]!=A[i,1] ,abs(dt)>=tol)):
            if A[i,1]==1:
                par=(lc1[0][int(A[i,2])],lc1[1][int(A[i,2])],lc1[2][int(A[i,2])],np.NaN,np.NaN,np.NaN)
            if A[i,1]==2:
                par=(np.NaN,np.NaN,np.NaN,lc2[0][int(A[i,2])],lc2[1][int(A[i,2])],lc2[2][int(A[i,2])])
            fin=np.vstack((fin,par))
    x = fin[~np.isnan(fin[:,0])]
    y = fin[~np.isnan(fin[:,3])]
    return A,fin
