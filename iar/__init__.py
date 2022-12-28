"""
Data sets, functions and scripts with examples to implement autoregressive models for irregularly observed time series. The models available in this package are the irregular autoregressive model (Eyheramendy et al.(2018) <doi:10.1093/mnras/sty2487>), the complex irregular autoregressive model (Elorrieta et al.(2019) <doi:10.1051/0004-6361/201935560>) and the bivariate irregular autoregressive model.
https://github.com/felipeelorrieta/iar
"""

__version__ = '1.2.8'

from .data_iar import clcep,eb,dmcep,dscut,agn,Planets,cvnovag,cvnovar
from .utils import gentime,harmonicfit,foldlc,pairingits
from .IARModel import IARsample,IARphiloglik,IARloglik,IARphikalman,IARkalman,IARfit,IARforecast,IARgsample,IARphigamma,IARgamma,IARtsample,IARphit,IARt,kde_sklearn,IARtest,IARpermutation,IARinterpolation,IARphikalman2,IARginterpolation,IARphigamma2
from .CIARModel import CIARsample,CIARphikalman,CIARkalman,CIARfit,CIARforecast,CIARinterpolation,CIARphikalman2
from .BIARModel import BIARsample,BIARphikalman,BIARkalman,BIARfit,BIARforecast,BIARinterpolation,BIARLL
