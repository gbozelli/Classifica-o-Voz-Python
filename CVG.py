import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def sigmoidFunction(Theta0,Theta1,X):
  YPred = 1/(1+np.exp(-(Theta1*X + Theta0)))
  return YPred

def DescentGradient(X,Y):
    Theta1, Theta0 = 0,0
    k,lam = 2000, 10
    N = len(X)
    for i in range(k):
        YPred = sigmoidFunction(Theta0,Theta1,X)
        Theta1 += lam*(2/N)*sum((Y-YPred)*X)
        Theta0 += lam*(2/N)*sum((Y-YPred))
        Error = 1/N*(sum((Y-YPred)**2))  
    return Theta1, Theta0, Error

def evaluate(xTest,yTest,Theta0,Theta1):
  label = []
  TN,TP,FN,FP = 0,0,0,0
  alpha=0.3
  for i in range(lenTrain,lenTrain+len(xTest)):
    if sigmoidFunction(Theta0,Theta1,xTest[i]) <= 0.5:
      label.append(0)
    else:
      label.append(1)
  for i in range(len(yTest)):
    if label[i] == yTest[lenTrain+i]:
      if label[i] == 0:
        TN += 1
        plt.scatter(xTest[lenTrain+i],label[i],color='red',alpha=alpha)
      if label[i] == 1:
        TP += 1
        plt.scatter(xTest[lenTrain+i],label[i],color='blue',alpha=alpha)
    if label[i] != yTest[lenTrain+i]:
      if label[i] == 0:
        FN += 1
        plt.scatter(xTest[lenTrain+i],label[i],color='red',alpha=alpha)
      if label[i] == 1:
        FP += 1
        plt.scatter(xTest[lenTrain+i],label[i],color='blue',alpha=alpha)
  accuracy, recall = TP/(TP+FP), TP/(TP+FN)
  fScore = (2*accuracy*recall)/(accuracy+recall)
  return fScore


filename = 'dadovozgenero.csv'
data = pd.read_csv(filename)
data1 = data.sample(frac = 1)
data1.reset_index(inplace = True)
x, y= data1['meanfun'], data1['label']
lenTrain = int(len(x)*0.8)
xTrain, xTest, yTrain, yTest = x[:lenTrain], x[lenTrain:], y[:lenTrain], y[lenTrain:]

Theta1, Theta0, Error = DescentGradient(xTrain,yTrain)
fScore = evaluate(xTest,yTest,Theta0,Theta1)

print(fScore)
X = np.arange(0.05,0.25,0.001)
positive = mpatches.Patch(color='red', label='Positivos')
negative = mpatches.Patch(color='blue', label='Negativos')
plt.plot(X,1/(1+np.exp(-(Theta1*X + Theta0))),color='black',label='Função Sigmóide')
plt.vlines(x=0.142,color='purple',ymin=0,ymax=1,linestyles='dotted',label='Limiar')
plt.legend(loc='upper right')
plt.legend(handles=[positive,negative])
print(Theta1,Theta0)
plt.grid(True)


