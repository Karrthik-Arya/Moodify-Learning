

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_json('https://raw.githubusercontent.com/Karrthik-Arya/Moodify-Learning/master/Assignment-2/data.json')
data.columns = ['x', 'y', 'label']

data.head()

data['label'].value_counts()

def plot_scatter_shape(data, cols, shape_col = 'label', col_y = 'y', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    unique_cats = data[shape_col].unique()
    for col in cols: # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            temp = data[data[shape_col] == cat]
            sns.regplot(x=col, y=col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.legend()
        plt.show()
plot_scatter_shape(data, ['x'], alpha = 0.05)

data_test = data.sample(frac = 0.2, random_state=200)
data_train = data.drop(data_test.index)
print(data_test.shape)
print(data_train.shape)
data_train.reset_index(drop = True, inplace= True)
data_test.reset_index(drop = True, inplace= True)

def fn(x,y,coef):
  fx = 0
  for i in range(5):
    fx += coef[i]*(x**(4-i))*(y**i)
  return fx
coef = [0.005,0.005,0.005,0.005,0.005]
results = fn(data_train['x'], data_train['y'], coef)
print(results)
results = 1/(1+np.exp(-1*results))
print(results)

loss = -data_train['label']*np.log(results) - (1-data_train['label'])*np.log(1-results)
loss = loss.sum()


def sgn(a):
  b = abs(a)
  return a/b

def grada1(data, results, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X**4)+((1-l)/(1-val))*(X**4)
  return grad.sum()
def grada2(data, results, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X**3*Y)+((1-l)/(1-val))*(X**3*Y)
  return grad.sum()
def grada3(data, results, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X**2*Y**2)+((1-l)/(1-val))*(X**2*Y**2)
  return grad.sum()
def grada4(data, results, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(X*(Y**3))+((1-l)/(1-val))*(X*(Y**3))
  return grad.sum()
def grada5(data, results, coef ):
  X = data['x']
  Y = data['y']
  l = data['label']
  val = coef[0]*X**4+coef[1]*(X**3)*Y+coef[2]*(X**2)*(Y**2)+coef[3]*(X**1)*(Y**3)+coef[4]*(Y**4)
  grad = -(l/val)*(Y**4)+((1-l)/(1-val))*(Y**4)
  return grad.sum()
  

for i in range(250):
  coef[0] = coef[0] - 0.0001*sgn(grada1(data, results, coef))
  coef[1] = coef[1] - 0.0001*sgn(grada2(data, results, coef))
  coef[2] = coef[2] - 0.0001*sgn(grada3(data, results, coef))
  coef[3] = coef[3] - 0.0001*sgn(grada4(data, results, coef))
  coef[4] = coef[4] - 0.0001*sgn(grada5(data, results, coef))
  results = fn(data_train['x'], data_train['y'], coef)
  results = 1/(1+np.exp(-1*results))
loss = -data_train['label']*np.log(results) - (1-data_train['label'])*np.log(1-results)
loss = loss.sum()
print(loss)


def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:]])
print(results.shape)
scores = score_model(results, 0.514)
unique, counts = np.unique(scores, return_counts=True)
print(np.asarray((unique, counts)))
data_train['label'].value_counts()

results = fn(data_test['x'], data_test['y'], coef)
results = 1/(1+np.exp(-1*results))
print(results)
loss = -data_train['label']*np.log(results) - (1-data_train['label'])*np.log(1-results)
loss = loss.sum()
print(loss)
scores = score_model(results, 0.514)
unique, counts = np.unique(scores, return_counts=True)
print(np.asarray((unique, counts)))
data_test['label'].value_counts()
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


    
print_metrics(data_test['label'], scores)



