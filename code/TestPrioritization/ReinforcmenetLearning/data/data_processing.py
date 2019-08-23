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
try:
	import cPickle as pickle
except:
	import pickle
    


    
if __name__ == '__main__':
    
    
    #data = pd.read_csv('execution_data.csv', header = 0, nrows = 15)
    data = pd.read_csv('commons_lang.csv', header = 0)
    data = data.rename(index=str, columns={"failures": "failures_4"})
    
    #cumulative_data = pd.read_csv('cumulative_data.csv', header = 0)
    time_median = data['time'].median()
    print(time_median)
    metric_list = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential', 'AvgLine', 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'CountDeclClass', 'CountDeclClassMethod', 'CountDeclClassVariable', 'CountDeclExecutableUnit', 'CountDeclFunction', 'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 'CountDeclMethod',	'CountDeclMethodDefault',	'CountDeclMethodPrivate', 'CountDeclMethodProtected', 'CountDeclMethodPublic', 'CountLine', 'CountLineBlank',	'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment', 'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic', 'MaxCyclomaticModified',	'MaxCyclomaticStrict', 'MaxEssential', 'MaxNesting', 'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential']
    #print(len(metric_list))
    
    data.insert(5,'time_0', 0, allow_duplicates = True)
    for i in range(0,4):
        data.insert(3+i,'failures_' + str(i), 0, allow_duplicates = True)    
    #print(data)
    
    cumulative_data = pd.DataFrame(columns = data.columns)
    result_data = pd.DataFrame(columns = data.columns)
    
    A_priority_array = []
    A_priority_arrary_with_time = []
    B_priority_array = []
    B_priority_arrary_with_time = []
    C_priority_array = []
    C_priority_arrary_with_time = []
    D_priority_array = []
    for i in range(0,len(data)):
        
        if (i%1000 == 0):
            print(i)
        x = cumulative_data.loc[cumulative_data['test_class_name'] == data.iloc[i]['test_class_name']]
        #x = cumulative_data.loc[cumulative_data['test_class_name'].isin(list(data.iloc[i]['test_class_name']))]
        #print(x)
        if x.empty:
            #print('EMPTY') #devo fare l'append della riga i-esima
            cumulative_data = cumulative_data.append(data.iloc[i][:])
            #data.iloc[i, data.columns.get_loc('time')] = 0.0
            data.iloc[i, data.columns.get_loc('timestamp')] = 0.0
            result_data = result_data.append(data.iloc[i][:])
            
            #priorità di tipo A
            A_priority_array.append(1.0 if data.iloc[i]['failures_4'] > 0 else 0.0)
            if data.iloc[i]['failures_4'] > 0:
                if data.iloc[i]['time'] < time_median:
                    A_priority = 3.0
                else:
                    A_priority = 2.0     
            else:
                if data.iloc[i]['time'] < time_median:
                    A_priority = 1.0
                else:
                    A_priority = 0.0
            A_priority_arrary_with_time.append(A_priority)
            #print(cumulative_data)
            
            #priorità di tipo B
            if data.iloc[i]['failures_4'] > 0:
                B_priority = 5.0
            elif data.iloc[i]['failures_3'] > 0:
                B_priority = 4.0
            elif data.iloc[i]['failures_2'] > 0:
                B_priority = 3.0
            elif data.iloc[i]['failures_1'] > 0:
                B_priority = 2.0
            elif data.iloc[i]['failures_0'] > 0:
                B_priority = 1.0
            else:
                B_priority = 0.0
            B_priority_array.append(B_priority)
            
            #priorità di tipo B con tempo
            if data.iloc[i]['failures_4'] > 0:
                if data.iloc[i]['time'] < time_median:
                    B_priority = 11.0
                else:
                    B_priority = 10.0
            elif data.iloc[i]['failures_3'] > 0:
                if data.iloc[i]['time'] < time_median:
                    B_priority = 9.0
                else:
                    B_priority = 8.0
            elif data.iloc[i]['failures_2'] > 0:
                if data.iloc[i]['time'] < time_median:
                    B_priority = 7.0
                else:
                    B_priority = 6.0
            elif data.iloc[i]['failures_1'] > 0:
                if data.iloc[i]['time'] < time_median:
                    B_priority = 5.0
                else:
                    B_priority = 4.0
            elif data.iloc[i]['failures_0'] > 0:
                if data.iloc[i]['time'] < time_median:
                    B_priority = 3.0
                else:
                    B_priority = 2.0
            else:
                if data.iloc[i]['time'] < time_median:
                    B_priority = 1.0
                else:
                    B_priority = 0.0
            B_priority_arrary_with_time.append(B_priority)
            
            
            #priorità di tipo C
            if data.iloc[i]['failures_%'] > 0.75:
                C_priority = 4.0
            elif data.iloc[i]['failures_%'] > 0.5:
                C_priority = 3.0
            elif data.iloc[i]['failures_%'] > 0.25:
                C_priority = 2.0
            elif data.iloc[i]['failures_%'] > 0.0:
                C_priority = 1.0
            else:
                C_priority = 0.0
            C_priority_array.append(C_priority)
            
            
            #priorità di tipo C con tempo
            if data.iloc[i]['failures_%'] > 0.75:
                if data.iloc[i]['time'] < time_median:
                    C_priority = 9.0
                else:
                    C_priority = 8.0
            elif data.iloc[i]['failures_%'] > 0.5:
                if data.iloc[i]['time'] < time_median:
                    C_priority = 7.0
                else:
                    C_priority = 6.0
            elif data.iloc[i]['failures_%'] > 0.25:
                if data.iloc[i]['time'] < time_median:
                    C_priority = 5.0
                else:
                    C_priority = 4.0
            elif data.iloc[i]['failures_%'] > 0.0:
                if data.iloc[i]['time'] < time_median:
                    C_priority = 3.0
                else:
                    C_priority = 2.0
            else:
                if data.iloc[i]['time'] < time_median:
                    C_priority = 1.0
                else:
                    C_priority = 0.0
            C_priority_arrary_with_time.append(C_priority)
            
            
            #priorità di tipo D
            if data.iloc[i]['failures_4'] > 0 or data.iloc[i]['time'] < time_median:
                D_priority = 1.0
            elif data.iloc[i]['failures_4'] == 0 or data.iloc[i]['time'] > time_median:
                D_priority = 0.0
            D_priority_array.append(D_priority)    
            
        
        else:
            
            #aggiorno le metriche in x
            for metric in metric_list:
                x.iloc[0, x.columns.get_loc(metric)] = data.iloc[i, data.columns.get_loc(metric)]
            
            #aggiorno la colonna cycle_id di x
            x.iloc[0, x.columns.get_loc('cycle_id')] = data.iloc[i, data.columns.get_loc('cycle_id')]
            #aggiorno la colonna tests di x
            x.iloc[0, x.columns.get_loc('tests')] = data.iloc[i, data.columns.get_loc('tests')]
            #aggiorno la colonna failures_% di x
            x.iloc[0, x.columns.get_loc('failures_%')] = data.iloc[i, data.columns.get_loc('failures_%')]
            #aggiorno la colonna time di x
            time_ring = collections.deque(maxlen=2)
            time_ring.append(x.iloc[0]['time_0'])
            time_ring.append(x.iloc[0]['time'])
            time_ring.append(data.iloc[i]['time'])
            x.iloc[0, x.columns.get_loc('time_0')] = time_ring[0]
            x.iloc[0, x.columns.get_loc('time')] = time_ring[1]
            #x.iloc[0, x.columns.get_loc('time')] = data.iloc[i, data.columns.get_loc('time')]
            #trasformo il timestamp in time_since di x
            x.iloc[0, x.columns.get_loc('timestamp')] = (data.iloc[i, data.columns.get_loc('timestamp')] - x.iloc[0, x.columns.get_loc('timestamp')])/60
            
            #aggiorno la storia dei fallimenti di x
            failures_ring = collections.deque(maxlen=5)
            for j in range(0, 5):   #le ultime posizioni contegono i verdetti (nel caso specifico le ultime 4)
                failures_ring.append(x.iloc[0]['failures_' + str(j)])
            #failures_ring.append(float(1))
            failures_ring.append(data.iloc[i]['failures_4'])
            #print(failures_ring)
            for j in range(0, 5):
                #x.iloc[0, x.columns.get_loc('failures_' + str(j+1))] = failures_ring[j]
                x.iloc[0, x.columns.get_loc('failures_' + str(j))] = failures_ring[j]
            
            
            #appendo x nel dataset dei dati definitivi
            #print(x)
            result_data = result_data.append(x, ignore_index = True)
            x.iloc[0, x.columns.get_loc('timestamp')] = data.iloc[i, data.columns.get_loc('timestamp')]
            #aggiorno cumulative-data
            cumulative_data.update(x)
            
            
            #priorità di tipo A
            A_priority_array.append(1.0 if x.iloc[0]['failures_4'] > 0 else 0.0)
            if x.iloc[0]['failures_4'] > 0:
                if x.iloc[0]['time'] < time_median:
                    A_priority = 3.0
                else:
                    A_priority = 2.0    
            else:
                if x.iloc[0]['time'] < time_median:
                    A_priority = 1.0
                else:
                    A_priority = 0.0
            
            A_priority_arrary_with_time.append(A_priority)
            #print(cumulative_data)
            
            
            #priorità di tipo B
            if x.iloc[0]['failures_4'] > 0:
                B_priority = 5.0
            elif x.iloc[0]['failures_3'] > 0:
                B_priority = 4.0
            elif x.iloc[0]['failures_2'] > 0:
                B_priority = 3.0
            elif x.iloc[0]['failures_1'] > 0:
                B_priority = 2.0
            elif x.iloc[0]['failures_0'] > 0:
                B_priority = 1.0
            else:
                B_priority = 0.0
            B_priority_array.append(B_priority)
            
            #priorità di tipo B con tempo
            if x.iloc[0]['failures_4'] > 0:
                if x.iloc[0]['time'] < time_median:
                    B_priority = 11.0
                else:
                    B_priority = 10.0
            elif x.iloc[0]['failures_3'] > 0:
                if x.iloc[0]['time'] < time_median:
                    B_priority = 9.0
                else:
                    B_priority = 8.0
            elif x.iloc[0]['failures_2'] > 0:
                if x.iloc[0]['time'] < time_median:
                    B_priority = 7.0
                else:
                    B_priority = 6.0
            elif x.iloc[0]['failures_1'] > 0:
                if x.iloc[0]['time'] < time_median:
                    B_priority = 5.0
                else:
                    B_priority = 4.0
            elif x.iloc[0]['failures_0'] > 0:
                if x.iloc[0]['time'] < time_median:
                    B_priority = 3.0
                else:
                    B_priority = 2.0
            else:
                if x.iloc[0]['time'] < time_median:
                    B_priority = 1.0
                else:
                    B_priority = 0.0
            B_priority_arrary_with_time.append(B_priority)
            
            
            #priorità di tipo C
            if x.iloc[0]['failures_%'] > 0.75:
                C_priority = 4.0
            elif x.iloc[0]['failures_%'] > 0.5:
                C_priority = 3.0
            elif x.iloc[0]['failures_%'] > 0.25:
                C_priority = 2.0
            elif x.iloc[0]['failures_%'] > 0:
                C_priority = 1.0
            else:
                C_priority = 0.0
            C_priority_array.append(C_priority)
            
            
            #priorità di tipo c con tempo
            if x.iloc[0]['failures_%'] > 0.75:
                if x.iloc[0]['time'] < time_median:
                    C_priority = 9.0
                else:
                    C_priority = 8.0
            elif x.iloc[0]['failures_%'] > 0.5:
                if x.iloc[0]['time'] < time_median:
                    C_priority = 7.0
                else:
                    C_priority = 6.0
            elif x.iloc[0]['failures_%'] > 0.25:
                if x.iloc[0]['time'] < time_median:
                    C_priority = 5.0
                else:
                    C_priority = 4.0
            elif x.iloc[0]['failures_%'] > 0.0:
                if x.iloc[0]['time'] < time_median:
                    C_priority = 3.0
                else:
                    C_priority = 2.0
            else:
                if x.iloc[0]['time'] < time_median:
                    C_priority = 1.0
                else:
                    C_priority = 0.0
            C_priority_arrary_with_time.append(C_priority)
            
            
            #priorità di tipo D
            if x.iloc[0]['failures_4'] > 0 or x.iloc[0]['time'] < time_median:
                D_priority = 1.0
            elif x.iloc[0]['failures_4'] == 0 or x.iloc[0]['time'] > time_median:
                D_priority = 0.0
            D_priority_array.append(D_priority)        
                    
    
    #A_priority
    result_data.insert(len(result_data.columns),'A_priority', A_priority_array, allow_duplicates = True)
    #A_priority_with_time
    result_data.insert(len(result_data.columns),'A_priority_with_time', A_priority_arrary_with_time, allow_duplicates = True)
    #B_priority
    result_data.insert(len(result_data.columns),'B_priority', B_priority_array, allow_duplicates = True)
    #B_priority_with_time
    result_data.insert(len(result_data.columns),'B_priority_with_time', B_priority_arrary_with_time, allow_duplicates = True)
    #C_priority
    result_data.insert(len(result_data.columns),'C_priority', C_priority_array, allow_duplicates = True)
    #C_priority_with_time
    result_data.insert(len(result_data.columns),'C_priority_with_time', C_priority_arrary_with_time, allow_duplicates = True)
    #D_priority_with_time
    result_data.insert(len(result_data.columns),'D_priority', D_priority_array, allow_duplicates = True)
      
    #print(cumulative_data)   
    print('RESULT_DATA')
    #print(result_data)
    
    result_data = result_data.rename(index=str, columns={"timestamp": "time_since"})
    result_data = result_data.rename(index=str, columns={"failures_4": "current_failures"})

    result_data.to_csv('result_data.csv', index = False)
    #cumulative_data.to_csv('cumulative_data.csv', index = False)
    #print(result_data)
    
