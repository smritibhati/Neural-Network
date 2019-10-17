#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
df = pd.read_csv('Apparel/apparel-trainval.csv')
from sklearn.preprocessing import LabelEncoder


# In[273]:


import csv
import copy
import matplotlib.pyplot as plt


# In[36]:


data,testdata= np.split(df,[int(0.80*len(df))])
X = data[data.columns[1:785]]
Y = data[data.columns[0]]


# In[37]:


means = np.mean(X, axis = 0)
stdDev = np.std(X, axis = 0)
X = (X - means) / stdDev


# In[38]:


def onehotencode(names):
    uniquenames =len(np.unique(names))
    onehot= np.zeros((len(names),uniquenames))
    onehot[np.arange(len(names)),names]=1
    return onehot


# In[39]:


encoder = LabelEncoder()
Y = data['label'].values
encoder.fit(Y)
Y = encoder.transform(Y)
Y = onehotencode(Y)


# In[134]:


accu=[]   
def sigmoid(x):
    return 1.0/(1.0+ np.exp(-x))

def sigmoidderivative(x):
    return x * (1.0 - x)

def softmax(x):
    temp = np.exp(x- np.max(x, axis = 1, keepdims=True))
    temp2 = np.sum(temp, axis = 1, keepdims= True)
    return temp/temp2

def der_tanh( z):
    return (1 - (np.tanh(z)**2))


# In[175]:


layeraccu = []
numlayers =[]
count=1
def reLu(z):
    Z = copy.deepcopy(z)
    Z[Z < 0] = 0
    return Z
    
def der_reLu(z):
    Z = copy.deepcopy(z)
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


# In[237]:


class NeuralNetwork:
    def __init__(self, X, Y, hiddeninfo):
        self.n = len(hiddeninfo)
        self.inputlayersize  = X.shape[1]
        self.outputlayersize = df['label'].unique().size  
        self.allweights = [.01] *(self.n + 1)
        self.allactivations = [.01] *(self.n+1 )
        self.allz = [.01]*(self.n +1)
        
        self.allweights[0] =  np.random.rand(self.inputlayersize,hiddeninfo[0])
        
        for i in range (1,self.n+1):
            if i==self.n:
                weights= np.random.rand(hiddeninfo[self.n-1],self.outputlayersize)
            else:
                weights = np.random.rand(hiddeninfo[i-1],hiddeninfo[i])
            self.allweights[i]=weights 
    
    def forwardprop(self,X,Y):
        
        output = []
        self.allz[0] = np.dot(X,self.allweights[0])
        self.allactivations[0] = sigmoid(self.allz[0])
#         self.allbias = 
        
        for i in range(1, len(self.allweights)):
            self.allz[i] = np.dot(self.allactivations[i-1],self.allweights[i])
            self.allactivations[i] = sigmoid(self.allz[i])

#         print(self.allz[-1])
        self.op = softmax(self.allz[-1]);
        return self.op
#         print(self.op)


    def backprop(self,X,Y):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        deltas = [np.float128(1.0)] * len(self.allweights)
        dweights = [np.float128(1.0)] *len(self.allweights)
        
        deltas[-1] = - (Y -self.op)*der_tanh(self.allactivations[-1]);
        dweights[-1] = np.dot(self.allactivations[-2].T, deltas[-1])
        
        
        i = len(self.allweights)-2
        while i>0:
            deltas[i] = np.dot(deltas[i+1],self.allweights[i+1].T)*der_tanh(self.allactivations[i])
            dweights[i] = np.dot(self.allactivations[i-1].T, deltas[i])
            i-=1
        
        deltas[0] =  np.dot(deltas[1],self.allweights[1].T)*der_tanh(self.allactivations[0])
        dweights[0] = np.dot(X.T, deltas[0])

        for i in range(len(self.allweights)):
            self.allweights[i] = self.allweights[i] - .01 * dweights[i]
        
#         print (self.op)
#         print(dweights)


# In[242]:


epochaccu=[]
epochcount=[]
epoch=0
def error(Y, op):
    return np.sum((Y - op)**2)


# In[259]:


nn = NeuralNetwork(X,Y,[128])
epoch+=5
for i in range(20):
    s = 0
    for i in range(480):
#         print (s)
        op = nn.forwardprop(X.iloc[s:s+100],Y[s:s+100])
        oneY = Y[s:s+100]
        nn.backprop(X.iloc[s:s+100],oneY)
        s+=100
    
    opall = nn.forwardprop(X,Y)
#     print(error(Y,opall))


# In[189]:


XX = testdata[testdata.columns[1:785]]
YY = testdata[testdata.columns[0]]
means = np.mean(XX, axis = 0)
stdDev = np.std(XX, axis = 0)
XX = (XX - means) / stdDev


# In[190]:


encoder = LabelEncoder()
Yy = testdata['label'].values
encoder.fit(YY)
YY = encoder.transform(YY)
YY = onehotencode(YY)


# In[260]:


op = nn.forwardprop(XX,YY)


# In[261]:


correct=0
for i in range(len(op)):
    temp1 = np.argmax(YY[i])
    temp2 = np.argmax(op[i])
    if(temp1 == temp2):
        correct += 1

epochaccu.append(correct*1.0/len(op))
epochcount.append(epoch)
# print(correct*1.0/len(op))


# In[166]:


print("Acuuracies using sigmoid, tanh and relu as activation functions", accu)


# # Accuracies with increasing number of layers

# In[257]:


plt.plot(numlayers, layeraccu)
plt.show()


# # Accuracies with increasing numbers of epochs

# In[272]:


plt.plot(epochcount, epochaccu)
plt.show()


# # Prediction for testdata

# In[293]:


TEST = df = pd.read_csv('apparel-test.csv')
means = np.mean(TEST, axis = 0)
stdDev = np.std(TEST, axis = 0)
TEST = (TEST - means) / stdDev

softm = nn.forwardprop(TEST,[])
predictions = []
for i in range(len(op)):
    temp = np.argmax(op[i])*1.0
    predictions.append(temp)

np.savetxt("2018201093_predict.csv", predictions, delimiter=",")


# In[ ]:




