import numpy as np
import scipy
from scipy.optimize import minimize,minimize_scalar
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.neighbors import KernelDensity
from .utils import harmonicfit

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

def IARg_sample(phi,n,sT,sigma2,mu):
    d=np.diff(sT)
    y=np.zeros(n)
    y[0]=np.random.gamma(shape=1, scale=1, size=1)
    shape=np.zeros(n)
    scale=np.zeros(n)
    yhat=np.zeros(n)
    for i in range(n-1):
        phid=phi**(d[i])
        yhat[i+1]=mu+phid * y[i]
        gL = sigma2*(1-phid**(2))
        shape[i+1]=yhat[i+1]**2/gL
        scale[i+1]=(gL/yhat[i+1])
        y[i+1]=np.random.gamma(shape=shape[i+1], scale=scale[i+1], size=1)
    return y, sT

def IAR_phi_gamma(x,y,sT):
    mu=x[1]
    sigma=x[2]
    x=x[0]
    d=np.diff(sT)
    n=len(y)
    phi=x**d
    yhat=mu+phi*y[0:(n-1)]
    gL=sigma*(1-phi**2)
    beta=gL/yhat
    alpha=yhat**2/gL
    s1=np.sum(-alpha*np.log(beta) - scipy.special.gammaln(alpha) - y[1:n]/beta + (alpha-1) * np.log(y[1:n])) - y[0]
    s1=-s1
    return s1

def IAR_gamma(y,sT):
    aux=1e10
    value=1e10
    br=0
    for i in range(20):
        phi=np.random.uniform(0,1,1).mean()
        mu=np.mean(y)*np.random.uniform(0,1,1).mean()
        sigma=np.var(y)*np.random.uniform(0,1,1).mean()
        bnds = ((0, 0.9999), (0.0001, np.mean(y)),(0.0001, np.var(y)))
        out=minimize(IAR_phi_gamma,np.array([phi, mu, sigma]),args=(y,sT),bounds=bnds,method='L-BFGS-B')
        value=out.fun
        if aux > value:
            par=out.x
            aux=value
            br=br+1
        if aux <= value and br>5 and i>10:
            break
        #print br
    if aux == 1e10:
       par=np.zeros(3)
    return par[0],par[1],par[2],aux

def IARt_sample(phi,n,sT,sigma2,nu):
    d=np.diff(sT)
    y=np.zeros(n)
    y[0]=np.random.normal(loc=0, scale=1, size=1)
    yhat=np.zeros(n)
    for i in range(n-1):
        phid=phi**(d[i])
        yhat[i+1]=phid * y[i]
        gL = sigma2*(1-phid**(2))
        y[i+1]=np.random.standard_t(df=nu,size=1)*np.sqrt(gL*(nu-2)/nu)+yhat[i+1]
    return y, sT

def IAR_phi_t(x,y,sT,nu):
    sigma=x[1]
    x=x[0]
    d=np.diff(sT)
    n=len(y)
    phi=x**d
    yhat=phi*y[0:(n-1)]
    gL=sigma*(1-phi**2)*(nu-2)/nu
    cte=(n-1)*np.log((scipy.special.gamma((nu+1)/2)/(scipy.special.gamma(nu/2)*np.sqrt(nu*np.pi))))
    stand=((y[1:n]-yhat)/np.sqrt(gL))**2
    s1=np.sum(0.5*np.log(gL))
    s2=np.sum(np.log(1 + (1/nu)*stand))
    out=cte-s1-((nu+1)/2)*s2 -0.5*(np.log(2*np.pi) + y[0]**2)
    out=-out
    return out

def IAR_t(y,sT,nu):
    aux=1e10
    value=1e10
    br=0
    for i in range(20):
        phi=np.random.uniform(0,1,1)[0]
        sigma=np.var(y)*np.random.uniform(0,1,1)[0]
        nu=float(nu)
        bnds = ((0, 0.9999), (0.0001, 2*np.var(y)))
        out=minimize(IAR_phi_t,np.array([phi, sigma]),args=(y,sT,nu),bounds=bnds,method='L-BFGS-B')
        value=out.fun
        if aux > value:
            par=out.x
            aux=value
            br=br+1
        if aux <= value and br>5 and i>10:
            break
        #print br                                                                               
    if aux == 1e10:
        par=np.zeros(2)
    return par[0],par[1],aux

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def IAR_Test(y,sT,f,phi,plot=True,xlim=np.arange(-1,0.1,1),bw=0.15,nameP='output.pdf'):
    aux=np.arange(2.5,48,2.5)
    aux=np.hstack((-aux,aux))
    aux=np.sort(aux)
    f0=f*(1+aux/100)
    f0=np.sort(f0)
    l1=len(f0)
    bad=np.zeros(l1)
    m=y
    for j in range(l1):
        res,sT=harmonicfit(sT,m,f0[j])
        y=res/np.sqrt(np.var(res,ddof=1))
        res3=IAR_loglik(y,sT,0)
        bad[j]=res3
    mubf=np.mean(np.log(bad))
    sdbf=np.std(np.log(bad),ddof=1)
    z0=np.log(phi)
    pvalue=scipy.stats.norm.cdf(z0,mubf,sdbf)
    norm=np.hstack((mubf,sdbf))
    if plot==True:
       pdf = matplotlib.backends.backend_pdf.PdfPages(nameP) 
       fig = plt.figure()
       xs = np.linspace(xlim[0],xlim[1],1000)
       density = kde_sklearn(np.log(bad),xs,bandwidth=bw)
       plt.plot(xs,density)
       plt.axis([xlim[0],xlim[1], 0, np.max(density)+0.01,])                                                                                                                                            
       plt.plot(z0, np.max(density)/100, 'o')
       pdf.savefig(1)
       pdf.close()
    return phi,norm,z0,pvalue
