from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
import xml.etree.cElementTree as ET
#import understand
import sys
import subprocess
import shlex
import operator
import numpy as np
import csv
import collections
import pandas as pd
import os
import random
import time
from datetime import datetime
from sklearn import neural_network
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
try:
	import cPickle as pickle
except:
	import pickle


if __name__ == '__main__':
    args = sys.argv[1]
    
    data = pd.read_csv(args, header = 0)
    
    commit_id_list = list(data['cycle_id'])
    #print(cycle_id_list)
    commit_id_list = list(dict.fromkeys(commit_id_list))
    
    ranking_array = [] 
    for commit_id in commit_id_list:
        print(commit_id)
        data_subset = data.loc[data['cycle_id'] == commit_id]
        #print(len(data_subset))
        
        for i in range(len(data_subset),0,-1):
            ranking_array.append(i)

    data.insert(len(data.columns), 'ranking', ranking_array, allow_duplicates = True)
    data.to_csv('ranking.csv', index = False, header = True)