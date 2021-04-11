import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_json('https://raw.githubusercontent.com/Karrthik-Arya/Moodify-Learning/master/Assignment-2/input.json')
data.columns = ['x', 'y', 'label']

from train import coef
from train import fn
results = fn(data['x'], data['y'], coef)
results = 1/(1+np.exp(-1*results))
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:]])
scores = score_model(results, 0.514)
data['label'] = scores
print(data)