import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def sigmoidFunction(Theta0,Theta1,X):
  Y = 1/(1+np.exp(-(Theta1*X + Theta0)))
  return Y

def descentGradient(X,Y):
    Theta1, Theta0 = 0,0
    k,lam = 2000, 10
    N = len(X)
    for i in range(k):
        YPred = sigmoidFunction(Theta0,Theta1,X)
        Theta1 += lam*(2/N)*sum((Y-YPred)*X)
        Theta0 += lam*(2/N)*sum((Y-YPred))
        Error = 1/N*(sum((Y-YPred)**2))
    return Theta1, Theta0, Error

def evaluateModel(xTest,yTest,Theta0,Theta1):
  label = []
  TN,TP,FN,FP = [],[],[],[]
  alpha=0.3
  for i in range(lenTrain,lenTrain+len(xTest)):
    if sigmoidFunction(Theta0,Theta1,xTest[i]) <= 0.5:
      label.append(0)
    else:
      label.append(1)
  for i in range(len(yTest)):
    if label[i] == yTest[lenTrain+i]:
      if label[i] == 0:
        TN.append(xTest[lenTrain+i])
      if label[i] == 1:
        TP.append(xTest[lenTrain+i])
    if label[i] != yTest[lenTrain+i]:
      if label[i] == 0:
        FN.append(xTest[lenTrain+i])
      if label[i] == 1:
        FP.append(xTest[lenTrain+i])
  return TN,TP,FN,FP

def evaluateTreshold(xTest,yTest,threshold):
  label = []
  TN,TP,FN,FP = [],[],[],[]
  for i in range(lenTrain,lenTrain+len(xTest)):
    if sigmoidFunction(Theta0,Theta1,xTest[i]) <= sigmoidFunction(Theta0,Theta1,threshold):
      label.append(0)
    else:
      label.append(1)
  for i in range(len(yTest)):
    if label[i] == yTest[lenTrain+i]:
      if label[i] == 0:
        TN.append(xTest[lenTrain+i])
      if label[i] == 1:
        TP.append(xTest[lenTrain+i])
    if label[i] != yTest[lenTrain+i]:
      if label[i] == 0:
        FN.append(xTest[lenTrain+i])
      if label[i] == 1:
        FP.append(xTest[lenTrain+i])
  return TN,TP,FN,FP

def plotROC(xTest,yTest):
  T, F = [],[]
  for i in range(20,250):
    threshold = i*1e-3
    TN,TP,FN,FP = evaluateTreshold(xTest,yTest,threshold)
    TN,TP,FN,FP = len(TN),len(TP),len(FN),len(FP)
    TPR, FPR = TP/(TP+FN), FP/(FP+TN)
    T.append(TPR)
    F.append(FPR)
  plt.plot(F,T,color='blue',label='Curva ROC')
  x = np.arange(0,1,0.1)
  plt.plot(x,x,color='black',linestyle='dashdot',label='Pior caso')
  plt.legend(loc='upper right')
  plt.grid(True)
  plt.xlabel('FPR')
  plt.ylabel('TPR')

def fScore(TP,FP,FN):
  accuracy, recall = len(TP)/(len(TP)+len(FP)), len(TP)/(len(TP)+len(FN))
  fScore = (2*accuracy*recall)/(accuracy+recall)
  return fScore

def plotFxT(xTest,yTest):
  T, F = [],[]
  for i in range(200,2500):
    threshold = i*1e-4
    TN,TP,FN,FP = evaluateTreshold(xTest,yTest,threshold)
    if len(TN)!=0 and len(TP)!=0 and len(FN)!=0 and len(FP)!=0:
      fscore = fScore(TP,FP,FN)
      T.append(threshold)
      F.append(fscore)
  plt.plot(T,F,color='blue',label='F1 x Limiar')
  plt.legend(loc='upper right')
  plt.grid(True)
  plt.xlabel('Limiar')
  plt.ylabel('F1')

def plotClassifier(Theta1,Theta0,TP,TN,FP,FN):
  TP = TP+FN
  TN = TN+FP
  X = np.arange(0.05,0.25,0.001)
  Y = np.full((1,len(TP)),1)
  plt.scatter(TP,Y,color='blue',label="Postitvos",alpha=0.3)
  Y = np.full((1,len(TN)),0)
  plt.scatter(TN,Y,color='red',label="Negativos",alpha=0.3)
  plt.plot(X,1/(1+np.exp(-(Theta1*X + Theta0))),color='black',label='Função Sigmóide')
  plt.vlines(x=0.142,color='purple',ymin=0,ymax=1,linestyles='dotted',label='Limiar = 0.144')
  plt.plot(X,1/(1+np.exp(-(Theta1*X + Theta0))),color='black',label='F1 = '+str(fScore)[0:4],alpha=0)
  plt.legend(loc='upper right')
  plt.xlabel('Frequência Fundamental')
  print(Theta1,Theta0)
  plt.grid(True)

filename = 'dadovozgenero.csv'
data = pd.read_csv(filename)
data1 = data.sample(frac = 1)
data1.reset_index(inplace = True)
x, y= data1['meanfun'], data1['label']
lenTrain = int(len(x)*0.8)
xTrain, xTest, yTrain, yTest = x[:lenTrain], x[lenTrain:], y[:lenTrain], y[lenTrain:]

Theta1, Theta0, Error = descentGradient(xTrain,yTrain)