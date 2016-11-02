from csv import DictReader
from math import exp, copysign, log, sqrt
from datetime import datetime
import random

class FM_FTRL_machine(object):
    
    def __init__(self, D, alpha, beta,L1,L2, dropoutRate = 1.0):
        self.alpha = alpha              # learning rate parameter alpha
        self.beta = beta                # learning rate parameter beta
        self.L1 = L1                    # L1 regularizer for first order terms
        self.L2 = L2                    # L2 regularizer for first order terms
        self.dropoutRate = dropoutRate  # dropout rate (which is actually the inclusion rate), i.e. dropoutRate = .8 indicates a probability of .2 of dropping out a feature.
        self.D = D
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * (D+1)  
        self.z = [0.] * (D+1) 
        self.w = [0.] * (D+1) 
        
    def predict_raw(self, x):
        ''' predict the raw score prior to logit transformation.
        '''
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        n = self.n
        z = self.z
        w = self.w
        raw_y = 0.
        for i in range (len(x)):
            sign = -1. if z[x[i]] < 0. else 1. # get sign of z[i]
            if sign * z[x[i]] <= L1:
                w[x[i]] = 0.
            else:
                w[x[i]] = (sign * L1 - z[x[i]]) / ((beta + sqrt(n[x[i]])) / alpha + L2)
            raw_y += w[x[i]]*1.0
        self.w = w
        return raw_y
    
    def predict(self, x):
        ''' predict the logit
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x), 35.), -35.)))
    
    def dropout(self, x):
        ''' dropout variables in list x
        '''
        for i, var in enumerate(x):
            if random.random() > self.dropoutRate:
                del x[i]
    
    def dropoutThenPredict(self, x):
        ''' first dropout some variables and then predict the logit using the dropped out data.
        '''
        self.dropout(x)
        return self.predict(x)
    
    def predictWithDroppedOutModel(self, x):
        ''' predict using all data, using a model trained with dropout.
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x) * self.dropoutRate, 35.), -35.)))
    
    def update(self, x, p, y):
        alpha = self.alpha
        w = self.w
        n = self.n
        z = self.z
        len_x = len(x)
        for i in range(len_x):
            g = (p - y)*1.0
            sigma = (sqrt(n[x[i]] + g * g) - sqrt(n[x[i]])) / self.alpha
            self.z[x[i]] += g - sigma * w[x[i]]
            self.n[x[i]] += g * g
    
    def write_w(self, filePath):
        ''' write out the first order weights w to a file.
        '''
        with open(filePath, "w") as f_out:
            for i, w in enumerate(self.w):
                f_out.write("%i,%f\n" % (i, w))
    
def logLoss(p, y):
    ''' 
    calculate the log loss cost
    p: prediction [0, 1]
    y: actual value {0, 1}
    '''
    p = max(min(p, 1. - 1e-15), 1e-15)
    return - log(p) if y == 1. else -log(1. - p)

def data(filePath, hashSize, hashSalt):
    ''' generator for data using hash trick
    
    INPUT:
        filePath
        hashSize
        hashSalt: String with which to salt the hash function
    '''
    
    for t, row in enumerate(DictReader(open(filePath))):
        ID = row['id']
        del row['id']
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']
        date = int(row['hour'][4:6])
        row['hour'] = row['hour'][6:]
        x = []
        for key in row:
            value = row[key]
            index = abs(hash(hashSalt + key + '_' + value)) % hashSize + 1      # 1 is added to hash index because I want 0 to indicate the bias term.
            x.append(index)
        
        yield t, date, ID, x, y
