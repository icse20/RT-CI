from sklearn.model_selection import train_test_split
import xml.etree.cElementTree as ET
import understand
import sys
import subprocess
import shlex
import operator
import numpy as np
import csv
import collections
import pandas as pd
import os
#import random
import time
from datetime import datetime
from sklearn import neural_network
try:
	import cPickle as pickle
except:
	import pickle
    
    

if __name__ == '__main__':
    
    data = pd.read_csv('commons_lang.csv', header = 0, nrows = 50)
    #print(data)
    
    commit_id_list = list(data['cycle_id'])
    #print(cycle_id_list)
    commit_id_list = list(dict.fromkeys(commit_id_list))
    #print(cycle_id_list)
    #print(len(commit_id_list))
    
    output_dataset = pd.DataFrame()
    for commit_id in commit_id_list:
        data_subset = data.loc[data['cycle_id'] == commit_id]
        
        #random_index = np.random.randint(0, len(data_subset))
        #print('INDEX')
        #print(random_index)
        if np.random.random() < 0.5:
            random_index = np.random.randint(0, len(data_subset))
            tests_no  = data_subset.iloc[random_index, data_subset.columns.get_loc('tests')]
            data_subset.iloc[random_index, data_subset.columns.get_loc('failures')] = np.random.randint(0,tests_no)
            data_subset.iloc[random_index, data_subset.columns.get_loc('failures_%')] = data_subset.iloc[random_index, data_subset.columns.get_loc('failures')] / data_subset.iloc[random_index, data_subset.columns.get_loc('tests')]
            
            random_index_2 = [np.random.randint(0, len(data_subset)) for i in range(0, np.random.randint(0, len(data_subset)))]
            print(random_index_2)
            for i in random_index_2:
            #for i in range(0, np.random.randint(0, len(data_subset)-1)):
                if np.random.random() < 0.3 and i != random_index:
                    tests_no_2  = data_subset.iloc[i, data_subset.columns.get_loc('tests')]
                    data_subset.iloc[i, data_subset.columns.get_loc('failures')] = np.random.randint(0,tests_no_2)
                    data_subset.iloc[i, data_subset.columns.get_loc('failures_%')] = data_subset.iloc[i, data_subset.columns.get_loc('failures')] / data_subset.iloc[i, data_subset.columns.get_loc('tests')]
            output_dataset = output_dataset.append(data_subset)
        else:
             output_dataset = output_dataset.append(data_subset)   
        print(data_subset)
    print('DEFINITIVE')
    print(output_dataset)
    
    #output_dataset.to_csv('commons_lang_injected.csv', index = False)
    