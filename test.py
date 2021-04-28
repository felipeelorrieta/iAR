from iar import IAR_sample,IAR_phi_loglik,IAR_loglik,gentime
import numpy as np
np.random.seed(6713)
sT=gentime(n=100)
y,sT =IAR_sample(0.99,100,sT)

phi=IAR_loglik(y,sT,0)
phi
