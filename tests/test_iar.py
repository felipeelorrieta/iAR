from iar import IARsample,IARphiloglik,IARloglik,gentime
import numpy as np
np.random.seed(6713)
sT=gentime(n=100)
y,sT =IARsample(0.99,100,sT)

phi=IARloglik(y,sT,0)
phi
