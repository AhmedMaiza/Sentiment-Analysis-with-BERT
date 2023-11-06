!pip install transformers

!pip install -q -U watermark

%reload_ext watermark
%watermark -v -p numpy,pandas,torch,transformers

import torch
from transformers import BertForSequenceClassification, BertModel, AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from tqdm import tqdm


from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF000D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize']=12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

sns.countplot(x='score', data=df)
plt.xlabel('review score');

def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  else:
    return 1

df.drop(df[df.score == 3].index, inplace=True)
df['sentiment'] = df.score.apply(to_sentiment)
class_names = ['negative', 'positive']
ax = sns.countplot(x='sentiment', data=df)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names);