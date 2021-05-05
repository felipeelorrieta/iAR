"""
Data sets, functions and scripts with examples to implement autoregressive models for irregularly observed time series. The models available in this package are the irregular autoregressive model (Eyheramendy et al.(2018) <doi:10.1093/mnras/sty2487>), the complex irregular autoregressive model (Elorrieta et al.(2019) <doi:10.1051/0004-6361/201935560>) and the bivariate irregular autoregressive model.
https://github.com/felipeelorrieta/iar
"""

__version__ = '1.0.0'

from .data_iar import clcep,eb,dmcep,dscut,agn,Planets
from .utils import gentime,harmonicfit,foldlc
from .IARModel import IAR_sample,IAR_phi_loglik,IAR_loglik,IAR_phi_kalman,IAR_kalman,IARg_sample,IAR_phi_gamma,IAR_gamma,IARt_sample,IAR_phi_t,IAR_t,kde_sklearn,IAR_Test
from .CIARModel import CIAR_sample,CIAR_phi_kalman,CIAR_kalman,CIAR_fit,CIAR_forecast
from .BIARModel import BIAR_sample,BIAR_phi_kalman,BIAR_kalman,BIAR_fit,BIAR_LL,BIAR_LL_Smoothing
