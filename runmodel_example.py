from FM_machine import *
import random
from math import log

random.seed(60)  # seed random variable for reproducibility
####################
#### PARAMETERS ####
####################
reportFrequency = 100000
trainingFile = "train.csv"

fm_dim = 1
fm_initDev = .01
hashSalt = "lucky"
    
alpha = .1
beta = 1.

alpha_fm = .01
beta_fm = 1.

p_D = 24
D = 2 ** p_D
L1 = 0.3
L2 = .1
n_epochs = 6

# initialize a FM learner
learner = FM_FTRL_machine(D, alpha, beta ,L1,L2)
for e in range(n_epochs):
    progressiveLoss = 0.
    progressiveCount = 0.
    for t, date, ID, x, y in data(trainingFile, D, hashSalt):
        p = learner.predict(x)
        loss = logLoss(p, y)
        learner.update(x, p, y)
        progressiveLoss += loss
        progressiveCount += 1.
        if t % reportFrequency == 0:                
            print("Epoch %d\tcount: %d\tProgressive Loss: %f" % (e, t, progressiveLoss / progressiveCount))
