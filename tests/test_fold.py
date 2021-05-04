from iar import data_iar,foldlc
import pandas as pd
data=data_iar.clcep()
t=data["t"]
m=data["m"]
merr=data["merr"]
f1=0.060033386
dataf=foldlc(t,m,merr,f1)
