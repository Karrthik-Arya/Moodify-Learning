
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_json('https://raw.githubusercontent.com/Karrthik-Arya/Moodify-Learning/master/Assignment-2/data.json')
data.columns = ['x', 'y', 'label']


data['label'].value_counts()

data_test = data.sample(frac = 0.2, random_state=200)
data_train = data.drop(data_test.index)
data_train.reset_index(drop = True, inplace= True)
data_test.reset_index(drop = True, inplace= True)

def fn(x,y,coef):
  fx = 0
  for i in range(5):
    fx += coef[i]*(x**(4-i))*(y**i)
  return fx
coef = [0.005,0.005,0.005,0.005,0.005]
results = fn(data_train['x'], data_train['y'], coef)

results = 1/(1+np.exp(-1*results))

loss = -data_train['label']*np.log(results) - (1-data_train['label'])*np.log(1-results)
loss = loss.sum()/4800
lr = 0.01

def grada1(data, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X**4)+((1-l)/(1-val))*(X**4)
  return grad.sum()/4800
def grada2(data, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X**3*Y)+((1-l)/(1-val))*(X**3*Y)
  return grad.sum()/4800
def grada3(data, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X**2*Y**2)+((1-l)/(1-val))*(X**2*Y**2)
  return grad.sum()/4800
def grada4(data, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X*(Y**3))+((1-l)/(1-val))*(X*(Y**3))
  return grad.sum()/4800
def grada5(data, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(Y**4)+((1-l)/(1-val))*(Y**4)
  return grad.sum()/4800
  


for i in range(750):
  coef[0] = coef[0] - lr*grada1(data, coef)
  coef[1] = coef[1] - lr*grada2(data, coef)
  coef[2] = coef[2] - lr*grada3(data, coef)
  coef[3] = coef[3] - lr*grada4(data, coef)
  coef[4] = coef[4] - lr*grada5(data, coef)
results = fn(data_train['x'], data_train['y'], coef)
results = 1/(1+np.exp(-1*results))
loss = -data_train['label']*np.log(results) - (1-data_train['label'])*np.log(1-results)
loss = loss.sum()/4800
print(loss)

def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:]])

scores = score_model(results, 0.56)

results = fn(data_test['x'], data_test['y'], coef)
results = 1/(1+np.exp(-1*results))
loss = -data_train['label']*np.log(results) - (1-data_train['label'])*np.log(1-results)
loss = loss.sum()/1200
print(loss)
scores = score_model(results, 0.56)

from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[1,1] + '             %5d' % conf[1,0])
    print('Actual negative    %6d' % conf[0,1] + '             %5d' % conf[0,0])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][1] + '        %6d' % metrics[3][0])
    print('Precision  %6.2f' % metrics[0][1] + '        %6.2f' % metrics[0][0])
    print('Recall     %6.2f' % metrics[1][1] + '        %6.2f' % metrics[1][0])
    print('F1         %6.2f' % metrics[2][1] + '        %6.2f' % metrics[2][0])


    
print_metrics(data_test['label'], scores)

print(coef)