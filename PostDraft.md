Text classification is a very classical problem. The goal is to classify documents into a fixed number of predefined categories, given a variable length of text bodies. It is widely use in sentimental analysis (IMDB, YELP reviews classification), stock market prediction, to GOOGLE's smart reply. This is a very active research area in academia. In the following discussion, I will try to present a few different approaches and compare their performances. Utimately, the goal for me is to implement the paper "Hierachical document classification". Given the limitation of data set I have, I will limite the excerise to Kaggle's IMDB dataset, and implementions are all based on Keras. 

Text classification using CNN

In the first series post, I will look into using convolutional neural network to solve this problem. 

import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

data_train = pd.read_csv('~/Testground/data/imdb/labeledTrainData.tsv', sep='\t')
print data_train.shape

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.sentiment[idx])
