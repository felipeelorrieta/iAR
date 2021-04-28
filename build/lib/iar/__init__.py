"""
Data sets, functions and scripts with examples to implement autoregressive models for irregularly observed time series. The models available in this package are the irregular autoregressive model (Eyheramendy et al.(2018) <doi:10.1093/mnras/sty2487>), the complex irregular autoregressive model (Elorrieta et al.(2019) <doi:10.1051/0004-6361/201935560>) and the bivariate irregular autoregressive model.
https://github.com/felipeelorrieta/iarPy
"""

__version__ = '1.0.0'


from .utils import gentime,harmonicfit
from .IARModel import IAR_sample,IAR_phi_loglik,IAR_loglik,IAR_phi_kalman,IAR_kalman
from .CIARModel import CIAR_sample,CIAR_phi_kalman,CIAR_kalman,CIAR_fit,CIAR_forecast
