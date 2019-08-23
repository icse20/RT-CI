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


#dimensione dell'i-esimo layer della rete, aumentando la lista delle dimensioni (e.g. 32,16,8,...) è possibile istanziare una rete con più layer
HIDDEN_LAYER_SIZES = 12,12,12,12
#REWARD_SELECTOR è utilizzato per selezionare la tipologia di reward desiderata, le scelte sono:
# A, A_WITH_TIME, B, B_WITH_TIME, C, C_WITH_TIME, D
REWARD_SELECTOR = 'A_WITH_TIME' #4 classi
#soglia temporale 
TIME_THRESHOLD = 0.006999999999999999
#dimensione della memoria utilizzata per il salvataggio dell'esperienza
MEMORY_SIZE = 2000
#dimensione del batch utilizzato per l'apprendimento, il batch viene dato in ingresso alla rete per il training della stessa
BATCH_SIZE = 2000
   
#La classe experience_replay è utilizzata per salvare l'esperienza. L'esperienza consiste nella tupla (stato, reward).
class experience_replay(object):
    def __init__(self, max_memory):
	    #max_memory è l'esperienza massima che può essere memorizzata
        self.max_memory = max_memory
        self.memory = self.build_memory()

    def build_memory(self):
        memory = collections.deque(maxlen = self.max_memory)
        
        return memory
        
        
	#Memorizza l'esperienza in memory
    #L'esperienza consiste nelle tuple (stato, reward)
    def remember(self, experience):
        self.memory.append(experience)

    #Il metodo get_batch è utilizzato per ritornare un batch di esperienza randomica di dimensione batch_size.
    def get_batch(self, batch_size):

        if batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
			#p è la probabilità di selezione di un'esperienza, la somma delle probabilità è 1
            batch_idx = np.random.choice(range(len(self.memory)), replace = False, size = batch_size, p = p)
            batch = [self.memory[idx] for idx in batch_idx]
        else:
            batch = self.memory
        return list(batch)
     
     
    
     
     #salva l'esperienza nella directory corrente  
    def save_memory(self, pkl_filename):
        with open(pkl_filename, 'wb') as file:  
            pickle.dump(self.memory, file)
             
             
     #carica l'esperienza
    def load_memory(self, pkl_filename):
        with open(pkl_filename, 'rb') as file:  
            pickle_memory = pickle.load(file)
        
        return pickle_memory
     
        
##################### AGENT START #####################        
class DQN_agent(object):
    def __init__(self, hidden_size, activation, warm_start, solver):
        self.hidden_size = hidden_size
        self.activation = activation
        self.warm_start = warm_start
        self.solver = solver
        self.already_fitted_model = False
        #self.experience = experience_replay(max_memory=500)
        self.model = self.build_model()
        
    def build_model(self):
        model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation=self.activation, warm_start=self.warm_start, solver=self.solver, max_iter=1500)
        
        return model
    
    def model_fitting(self, Xtrain, Ytrain):
        self.model.fit(Xtrain, Ytrain)
        
    def get_action(self, state):
        action = self.model.predict(np.array(state))
        action_p = self.model.predict_proba(np.array(state))
        
        return (action, action_p)
    
    #salva il modello nella directory corrente  
    def save_model(self, pkl_filename):
        with open(pkl_filename, 'wb') as file:  
            pickle.dump(self.model, file)
    
    #carica il modello
    def load_model(self, pkl_filename):
        with open(pkl_filename, 'rb') as file:  
            pickle_model = pickle.load(file)
        
        return pickle_model 
#####################################################


##################### REWARD ########################
#attribuisce una reward di 1 se c'è stato almeno un fallimento nella classe di test, 0 altrimenti
def A_reward(data):
    A_priority_array = []
    
    for i in range(0, len(data)):
        A_priority_array.append(1.0 if data.iloc[i]['current_failures'] > 0 else 0.0)
        
    data.insert(len(data.columns),'reward', A_priority_array, allow_duplicates = True)
    return(data)
 
#attribuisce una reward di 3 se c'è stato almeno un fallimento nella classe di test (current_failures > 0) e il tempo di esecuzione è minore di una soglia (exec_time < threshold)
#attribuisce una reward di 2 se c'è stato almeno un fallimento nella classe di test (current_failures > 0) e il tempo di esecuzione è maggiore di una soglia (exec_time > threshold)
#attribuisce una reward di 1 se non c'è stato alcun fallimento nelle classe dit test (current_failures = 0) e il tempo di esecuzione è minore di una soglia (exec_time < threshold)
#attribuisce una reward di 0 se non c'è stato alcun fallimento nelle classe dit test (current_failures = 0) e il tempo di esecuzione è maggiore di una soglia (exec_time > threshold)
def A_reward_with_time(data, threshold):  
    A_priority_arrary_with_time = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['current_failures'] > 0:
            if data.iloc[i]['time'] < threshold:
                priority = 3.0
            else:
                priority = 2.0     
        else:
            if data.iloc[i]['time'] < threshold:
                priority = 1.0
            else:
                priority = 0.0
        A_priority_arrary_with_time.append(priority)
        
    data.insert(len(data.columns),'reward', A_priority_arrary_with_time, allow_duplicates = True)
    return(data)



#attribuisce una reward di 4 se nell'esecuzione i-esima è stato rilevato un fallimento, altrimenti
    #attribuisce una reward di 3 se nell'esecuzione (i-1)-esima è stato rilevato un fallimento, altrimenti
        #attribuisce una reward di 2 se nell'esecuzione (i-2)-esima è stato rilevato un fallimento, altrimenti
            #attribuisce una reward di 1 se nell'esecuzione (i-3)-esima è stato rilevato un fallimento, altrimenti
                #attribuisce una reward di 0
def B_reward(data):
    B_priority_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['current_failures'] > 0:
            priority = 5.0
        elif data.iloc[i]['failures_3'] > 0:
            priority = 4.0
        elif data.iloc[i]['failures_2'] > 0:
            priority = 3.0
        elif data.iloc[i]['failures_1'] > 0:
            priority = 2.0
        elif data.iloc[i]['failures_0'] > 0:
            priority = 1.0
        else:
            priority = 0.0
        B_priority_array.append(priority)
        
    data.insert(len(data.columns),'reward', B_priority_array, allow_duplicates = True)
    return(data)


def B_reward_with_time(data, threshold):
    B_priority_arrary_with_time = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['current_failures'] > 0:
            if data.iloc[i]['time'] < threshold:
                priority = 11.0
            else:
                priority = 10.0
        elif data.iloc[i]['failures_3'] > 0:
            if data.iloc[i]['time'] < threshold:
                priority = 9.0
            else:
                priority = 8.0
        elif data.iloc[i]['failures_2'] > 0:
            if data.iloc[i]['time'] < threshold:
                priority = 7.0
            else:
                priority = 6.0
        elif data.iloc[i]['failures_1'] > 0:
            if data.iloc[i]['time'] < threshold:
                priority = 5.0
            else:
                priority = 4.0
        elif data.iloc[i]['failures_0'] > 0:
            if data.iloc[i]['time'] < threshold:
                priority = 3.0
            else:
                priority = 2.0
        else:
            if data.iloc[i]['time'] < threshold:
                priority = 1.0
            else:
                priority = 0.0
        B_priority_arrary_with_time.append(priority)
        
    data.insert(len(data.columns),'reward', B_priority_arrary_with_time, allow_duplicates = True)
    return(data)


    
#attribuisce una reward di 4 se nell'esecuzione i-esima la percentuale di testcase falliti sul totale è maggiore del 75% (failures_% > 0.75), altrimenti
    #attribuisce una reward di 3 se nell'esecuzione (i-1)-esima la percentuale di testcase falliti sul totale è maggiore del 50% (failures_% > 0.50), altrimenti
        #attribuisce una reward di 2 se nell'esecuzione (i-2)-esima la percentuale di testcase falliti sul totale è maggiore del 0.25% (failures_% > 0.25), altrimenti
            #attribuisce una reward di 1 se nell'esecuzione (i-3)-esima la percentuale di testcase falliti sul totale è maggiore dello 0% (failures_% > 0.0), altrimenti
                #attribuisce una reward di 0    
def C_reward(data):
    C_priority_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['failures_%'] > 0.75:
            priority = 4.0
        elif data.iloc[i]['failures_%'] > 0.5:
            priority = 3.0
        elif data.iloc[i]['failures_%'] > 0.25:
            priority = 2.0
        elif data.iloc[i]['failures_%'] > 0.0:
            priority = 1.0
        else:
            priority = 0.0
        C_priority_array.append(priority)
    
    data.insert(len(data.columns),'reward', C_priority_array, allow_duplicates = True)
    return(data)



#attribuisce una reward di 9 se nell'esecuzione i-esima la percentuale di testcase falliti sul totale è maggiore del 75% (failures_% > 0.75) e il tempo di esecuzione è minore di una soglia (exec_time < threshold), altrimenti
#attribuisce una reward di 8 se nell'esecuzione i-esima la percentuale di testcase falliti sul totale è maggiore del 75% (failures_% > 0.75) e il tempo di esecuzione è maggiore di una soglia (exec_time > threshold), altrimenti 
#e cosi vià
def C_reward_with_time(data, threshold):
    C_priority_arrary_with_time = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['failures_%'] > 0.75:
            if data.iloc[i]['time'] < threshold:
                priority = 9.0
            else:
                priority = 8.0
        elif data.iloc[i]['failures_%'] > 0.5:
            if data.iloc[i]['time'] < threshold:
                priority = 7.0
            else:
                priority = 6.0
        elif data.iloc[i]['failures_%'] > 0.25:
            if data.iloc[i]['time'] < threshold:
                priority = 5.0
            else:
                priority = 4.0
        elif data.iloc[i]['failures_%'] > 0.0:
            if data.iloc[i]['time'] < threshold:
                priority = 3.0
            else:
                priority = 2.0
        else:
            if data.iloc[i]['time'] < threshold:
                priority = 1.0
            else:
                priority = 0.0
        C_priority_arrary_with_time.append(priority)

    data.insert(len(data.columns),'reward', C_priority_arrary_with_time, allow_duplicates = True)
    return(data)


#attribuisce una reward di 1 se c'è stato almeno un fallimento nella classe di test (current_failures > 0) o il tempo di esecuzione è minore di una soglia (exec_time < threshold)
#attribuisce una reward di 0 se non c'è stato alcun fallimento nelle classe dit test (current_failures = 0) o il tempo di esecuzione è maggiore di una soglia (exec_time > threshold)
#In questo modo vado ad attribuire una reward maggiore ai casi di test che sono falliti o che durano poco o entrambe le cose 
def D_reward(data, threshold):
    D_priority_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['current_failures'] > 0 or data.iloc[i]['time'] <= threshold:
            priority = 1.0
        elif data.iloc[i]['current_failures'] == 0 or data.iloc[i]['time'] > threshold:
            priority = 0.0
        D_priority_array.append(priority)
        
    data.insert(len(data.columns),'reward', D_priority_array, allow_duplicates = True)
    return(data)
#####################################################    

    
############## SELETTORE DELLE REWARD ###############
#funzione utilizzata per attribuire la reward ai cicli di CI diversi dal primo 
def reward(REWARD_SELECTOR, data, threshold):  
    if REWARD_SELECTOR == 'A':
        rewarded_data = A_reward(data)
    elif REWARD_SELECTOR == 'A_WITH_TIME':
        rewarded_data = A_reward_with_time(data, threshold)
    elif REWARD_SELECTOR == 'B':
        rewarded_data = B_reward(data)
    elif REWARD_SELECTOR == 'B_WITH_TIME':
        rewarded_data = B_reward_with_time(data, threshold)
    elif REWARD_SELECTOR == 'C':
        rewarded_data = C_reward(data)
    elif REWARD_SELECTOR == 'C_WITH_TIME':
        rewarded_data = C_reward_with_time(data, threshold)
    elif REWARD_SELECTOR == 'D':
        rewarded_data = D_reward(data, threshold)
    else:
        print('INVALID REWARD TYPE')
    return(rewarded_data)
#####################################################################       


##################### FIRST CYCLE REWARD ########################

def A_first_cycle_reward(data):
    reward_array = []
    
    for i in range(0, len(data)):
        reward_array.append(1.0 if data.iloc[i]['priority'] >= 0.5 else 0.0)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)
    
    
def A_first_cycle_reward_with_time(data):
    reward_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['priority'] >= (1/4)*3:
            reward = 3.0
        elif data.iloc[i]['priority'] >= (1/4)*2:
            reward = 2.0
        elif data.iloc[i]['priority'] >= (1/4):
            reward = 1.0
        else:
            reward = 0.0
        reward_array.append(reward)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)


def B_first_cycle_reward(data):
    reward_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['priority'] >= (1/6)*5:
            reward = 5.0
        elif data.iloc[i]['priority'] >= (1/6)*4:
            reward = 4.0
        elif data.iloc[i]['priority'] >= (1/6)*3:
            reward = 3.0
        elif data.iloc[i]['priority'] >= (1/6)*2 :
            reward = 2.0
        elif data.iloc[i]['priority'] >= (1/6):
            reward = 1.0
        else:
            reward = 0.0
        reward_array.append(reward)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)
 
def B_first_cycle_reward_with_time(data):
    reward_array = []  
    
    for i in range(0, len(data)):
        if data.iloc[i]['priority'] >= (1/12)*11:
            reward = 11.0
        elif data.iloc[i]['priority'] >= (1/12)*10:
            reward = 10.0
        elif data.iloc[i]['priority'] >= (1/12)*9:
            reward = 9.0
        elif data.iloc[i]['priority'] >= (1/12)*8:
            reward = 8.0
        elif data.iloc[i]['priority'] >= (1/12)*7:
            reward = 7.0
        elif data.iloc[i]['priority'] >= (1/12)*6:
            reward = 6.0
        elif data.iloc[i]['priority'] >= (1/12)*5:
            reward = 5.0
        elif data.iloc[i]['priority'] >= (1/12)*4:
            reward = 4.0
        elif data.iloc[i]['priority'] >= (1/12)*3:
            reward = 3.0
        elif data.iloc[i]['priority'] >= (1/12)*2:
            reward = 2.0
        elif data.iloc[i]['priority'] >= (1/12):
            reward = 1.0
        else:
            reward = 0.0
        reward_array.append(reward)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)
    
    
def C_first_cycle_reward(data):
    reward_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['priority'] >= (1/5)*4:
            reward = 4.0
        elif data.iloc[i]['priority'] >= (1/5)*3:
            reward = 3.0
        elif data.iloc[i]['priority'] >= (1/5)*2:
            reward = 2.0
        elif data.iloc[i]['priority'] >= (1/5):
            reward = 1.0
        else:
            reward = 0.0
        reward_array.append(reward)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)
     
   
def C_first_cycle_reward_with_time(data):
    reward_array = []
    
    for i in range(0, len(data)):
        if data.iloc[i]['priority'] >= (1/10)*9:
            reward = 9.0
        elif data.iloc[i]['priority'] >= (1/10)*8:
            reward = 8.0
        elif data.iloc[i]['priority'] >= (1/10)*7:
            reward = 7.0
        elif data.iloc[i]['priority'] >= (1/10)*6:
            reward = 6.0
        elif data.iloc[i]['priority'] >= (1/10)*5:
            reward = 5.0
        elif data.iloc[i]['priority'] >= (1/10)*4:
            reward = 4.0
        elif data.iloc[i]['priority'] >= (1/10)*3:
            reward = 3.0
        elif data.iloc[i]['priority'] >= (1/10)*2:
            reward = 2.0
        elif data.iloc[i]['priority'] >= (1/10):
            reward = 1.0
        else:
            reward = 0.0
        reward_array.append(reward)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)
     
     
def D_first_cycle_reward(data):
    reward_array = []
    
    for i in range(0, len(data)):
        reward_array.append(1.0 if data.iloc[i]['priority'] >= 0.5 else 0.0)
        
    data.insert(len(data.columns),'reward', reward_array, allow_duplicates = True)
    return(data)
########################################################################    

    
############## SELETTORE DELLE REWARD PER IL PRIMO CICLO ###############
#funzione utilizzata per attribuire la reward al primo ciclo di CI
def first_cycle_reward(REWARD_SELECTOR, data):
    if REWARD_SELECTOR == 'A':
        rewarded_data = A_first_cycle_reward(data)
    elif REWARD_SELECTOR == 'A_WITH_TIME':
        rewarded_data = A_first_cycle_reward_with_time(data)
    elif REWARD_SELECTOR == 'B':
        rewarded_data = B_first_cycle_reward(data)
    elif REWARD_SELECTOR == 'B_WITH_TIME':
        rewarded_data = B_first_cycle_reward_with_time(data)
    elif REWARD_SELECTOR == 'C':
        rewarded_data = C_first_cycle_reward(data)
    elif REWARD_SELECTOR == 'C_WITH_TIME':
        rewarded_data = C_first_cycle_reward_with_time(data)
    elif REWARD_SELECTOR == 'D':
        rewarded_data = D_first_cycle_reward(data)
    else:
        print('INVALID REWARD TYPE')
    return(rewarded_data)    
########################################################################  

#FAILURE PERCENTILE ACCURACY (FPA) GENERATOR
def FPA_generator(evaluation):
    fpa = 0.0
    for m in range(1, len(evaluation)+1):
        ranking_sum = 0.0
        for i in range(0, m):
            ranking_sum = ranking_sum + evaluation.iloc[i]['ranking']
            if i == m - 1:
                ranking_sum = ranking_sum / evaluation['ranking'].sum()
        #print(ranking_sum)
        fpa = fpa + ranking_sum
    return(fpa / len(evaluation))
    


if __name__ == '__main__':
    #args[0] nome del file da processare
    args = sys.argv[1]
    print('\nDATASET')
    print(args + '\n')
    #leggo il dataset con i dati delle esecuzioni
    data = pd.read_csv(args, header = 0)
    #data = pd.read_csv('commons_lang_result.csv', header = 0, nrows = 100)
    data = data.rename(index=str, columns={"A_priority":"A" , "A_priority_with_time":"A_WITH_TIME", "B_priority":"B", "B_priority_with_time":"B_WITH_TIME", "C_priority":"C", "C_priority_with_time":"C_WITH_TIME", "D_priority":"D" })
    #print(data)
    labels = data[['cycle_id',REWARD_SELECTOR]]
    #print(labels)
    data = data.drop(['A', 'A_WITH_TIME', 'B', 'B_WITH_TIME', 'C', 'C_WITH_TIME', 'D'], axis = 'columns')
    #print(data.iloc[:,:12])
    
    
    #istanziazione agente
    agent = DQN_agent(HIDDEN_LAYER_SIZES, 'relu', False, 'adam')
    print(agent.model)
    #istanziazione memoria utilizzata per il salvataggio dell'esperienza
    mem = experience_replay(MEMORY_SIZE)
    
    #time_median = data['time'].median()
    #print(time_median)
    
    commit_id_list = list(data['cycle_id'])
    #print(cycle_id_list)
    commit_id_list = list(dict.fromkeys(commit_id_list))
    #print(cycle_id_list)
    
    #dataset dei risultati finali
    prediction_arry = []
    output_data = pd.DataFrame()
    prediction_time = []
    learning_time = []
    median = 0.0
    
    for commit_id in commit_id_list:
        print(commit_id)
        data_subset = data.loc[data['cycle_id'] == commit_id]
        
        
        #se already_fitted_model == True allora ho già addestrato la rete almeno una volta
        if agent.already_fitted_model:
            #print('ALL_THE_OTHER_CYCLE')
            #la priorità viene attribuita utilizzando la rete
            #prendo il tempo della predizione
            prediction_start = time.time()
            (action, action_p) = agent.get_action(data_subset.drop(['test_class_name', 'cycle_id', 'current_failures', 'time'], axis = 'columns'))
            prediction_end = time.time()  
            print('PREDICTION TIME')
            print(prediction_end - prediction_start)
            prediction_time.append(prediction_end - prediction_start)
            #print(action)
            #lista di predizioni utilizzata per calcolare precision e recall
            for elem in action:
                prediction_arry.append(elem)
            #creo dataframe con le probabilità delle azioni
            action_p = pd.DataFrame(action_p, columns = agent.model.classes_)
            #print(agent.model.classes_)
            
            #gestione della priorità
            priority_p_array = []
            for j in range(0, len(action)):
                priority_p_array.append(action_p.iloc[j][action[j]])
            #print(priority_p_array)
                
            #stampo l'accuracy per commit
            score = agent.model.score(data_subset.drop(['test_class_name', 'cycle_id', 'current_failures', 'time'], axis = 'columns'), labels.loc[labels['cycle_id'] == commit_id][REWARD_SELECTOR]) 
            print("Test score: {0:.2f} %".format(100 * score))
            
            #classification report
            #print(classification_report(labels.loc[labels['cycle_id'] == commit_id][REWARD_SELECTOR], action))
            #confusion matrix
            #cm = confusion_matrix(labels.loc[labels['cycle_id'] == commit_id][REWARD_SELECTOR], action)
            #print(cm)
            #confusion_matrix = pd.crosstab(labels.loc[labels['cycle_id'] == commit_id][REWARD_SELECTOR], action, rownames=['True'], colnames=['Predicted'], margins=True)
            #print(confusion_matrix)
            
            #print('LABELS')
            #print(labels.loc[labels['cycle_id'] == commit_id][REWARD_SELECTOR])
            #inserisco la colonna delle probabilità delle classi (priority_p)
            data_subset.insert(len(data_subset.columns),'priority_p', priority_p_array, allow_duplicates = True)
            #appendo la colonna delle classi (priority)
            data_subset.insert(len(data_subset.columns),'priority', action, allow_duplicates = True)
            #print(data_subset.iloc[:, -5:])
            #ordino in maniera decrescente in base alla colonna priority_p
            data_subset = data_subset.sort_values(by = 'priority_p', ascending = False)
            #print(data_subset.iloc[:, -5:])
            #ordino in maniera decrescente in base alla colonna priority
            data_subset = data_subset.sort_values(by = 'priority', ascending = False)
            #print(data_subset.iloc[:, -5:])
            #attribuisco la reward
            data_subset = reward(REWARD_SELECTOR, data_subset, median)
            #print(data_subset) #-DECOMMENT
            print('MEDIAN JUST USED')
            print(median)
            #print(data_subset['time'])
            #print(data_subset['reward'])
            
            ####################################################################################################################################################
            #VALUTAZIONE DELLE PRESTAZIONI
            evaluation = pd.DataFrame()
            evaluation.insert(len(evaluation.columns),'priority', action, allow_duplicates = True)
            evaluation.insert(len(evaluation.columns),'priority_p', priority_p_array, allow_duplicates = True)
            
            time_array = []
            class_array = []
            failures_array = []
            failures_percenteage_array = []
            for j in range(0, len(data_subset)):
                time_array.append(data_subset.iloc[j]['time'])
                class_array.append(data_subset.iloc[j]['reward'])
                failures_array.append(data_subset.iloc[j]['current_failures'])
                failures_percenteage_array.append(data_subset.iloc[j]['failures_%'])
            
            evaluation.insert(len(evaluation.columns), 'time', time_array, allow_duplicates = True)
            evaluation.insert(len(evaluation.columns), 'class', class_array, allow_duplicates = True)
            evaluation.insert(len(evaluation.columns), 'failures', failures_array, allow_duplicates = True)
            evaluation.insert(len(evaluation.columns), 'failures_percenteage', failures_percenteage_array, allow_duplicates = True)
            #print(data_subset.iloc[:,-5:])
            
            #ordino sul tempo e sulla classe
            evaluation = evaluation.sort_values(by = ['class', 'time'], ascending = [False, True])
            #print(evaluation)
            #inserisco la colonna RANKING
            rank = [i for i in range(len(evaluation), 0, -1)]
            evaluation.insert(len(evaluation.columns),'ranking', rank, allow_duplicates = True)
            #print(evaluation)
            
            #OPTIMAL FAILURE PERCENTILE ACCURACY (FPA) BEFORE RANKING
            print('OPTIMAL_FPA')
            optimal_fpa = FPA_generator(evaluation)
            print(optimal_fpa)
            #Calcolo tempo di esecuzione e fallimenti prima dell'ordinamento
            optimal_exec_time_25 = evaluation['time'].head(max(int(len(data_subset)/4), 1)).sum()
            optimal_exec_time_50 = evaluation['time'].head(max(int(len(data_subset)/4), 1)*2).sum()
            optimal_exec_time_75 = evaluation['time'].head(max(int(len(data_subset)/4), 1)*3).sum()
            optimal_failures_25 = evaluation['failures'].head(max(int(len(data_subset)/4), 1)).sum()
            optimal_failures_50 = evaluation['failures'].head(max(int(len(data_subset)/4), 1)*2).sum()
            optimal_failures_75 = evaluation['failures'].head(max(int(len(data_subset)/4), 1)*3).sum()
            
            
            #ordino su priority_p e priority
            evaluation = evaluation.sort_values(by = ['priority', 'priority_p'], ascending = [False, False])
            #print(evaluation)
            
            #Calcolo tempo di esecuzione e fallimenti dopo l'ordinamento
            exec_time_25 = evaluation['time'].head(max(int(len(data_subset)/4), 1)).sum()
            exec_time_50 = evaluation['time'].head(max(int(len(data_subset)/4), 1)*2).sum()
            exec_time_75 = evaluation['time'].head(max(int(len(data_subset)/4), 1)*3).sum()
            failures_in_25_ordered = evaluation['failures'].head(max(int(len(data_subset)/4), 1)).sum()
            failures_in_50_ordered = evaluation['failures'].head(max(int(len(data_subset)/4), 1)*2).sum()
            failures_in_75_ordered = evaluation['failures'].head(max(int(len(data_subset)/4), 1)*3).sum()
            
            #ESTIMATED FAILURE PERCENTILE ACCURACY (FPA) AFTER RANKING 
            print('ESTIMATED_FPA')   
            estimated_fpa = FPA_generator(evaluation)
            print(estimated_fpa)
            
            
            #metriche per ciclo
            output_data_temp = pd.DataFrame({"cycle_id":[commit_id], "num_testsuite":[len(data_subset)], "NORMALIZED_FPA":[estimated_fpa / optimal_fpa], "accuracy":[score], "total_failures_in_cycle":[evaluation['failures'].sum()], "exec_time":[evaluation['time'].sum()], "optimal_failures_25%":[optimal_failures_25], "failures_in_25%_ordered":[failures_in_25_ordered], "optimal_exec_time_25%":[optimal_exec_time_25], "exec_time_25%":[exec_time_25], "optimal_failures_50%":[optimal_failures_50], "failures_in_50%_ordered":[failures_in_50_ordered], "optimal_exec_time_50%":[optimal_exec_time_50],"exec_time_50%":[exec_time_50], "optimal_failures_75%":[optimal_failures_75], "failures_in_75%_ordered":[failures_in_75_ordered], "optimal_exec_time_75%":[optimal_exec_time_75], "exec_time_75%":[exec_time_75]})
            output_data = output_data.append(output_data_temp)
            ####################################################################################################################################################
            
            #elimino dal dataset le colonne relative ai fallimenti correnti e al tempo corrente
            data_subset = data_subset.drop(['current_failures', 'priority_p', 'priority'], axis = 'columns')
            #print(data_subset)
            #salvo l'esperienza in memoria utilizzando la classe 'experience_replay' 
            for i in range(0,len(data_subset)):
                mem.remember(list(data_subset.iloc[i]))
            
        else:
            #print('FIRST_CYCLE')
            #nel primo ciclo non ho a disposizione un modello addestrato, di conseguenza la priorità viene attribuita in maniera randomica tra 0 e 1
            priority = [np.random.random() for i in range(0,len(data_subset))]
            #appendo la colonna delle priorità
            data_subset.insert(len(data_subset.columns),'priority', priority, allow_duplicates = True)
            #ordino in maniera decrescente in base alla priorità
            data_subset = data_subset.sort_values(by = 'priority', ascending = False)
            #attribuisco la reward
            data_subset = first_cycle_reward(REWARD_SELECTOR, data_subset)
            #print(data_subset) #-DECOMMENT
            #elimino dal dataset le colonne relative ai fallimenti correnti e al tempo corrente
            data_subset = data_subset.drop(['current_failures', 'priority'], axis = 'columns')
            #salvo l'esperienza in memoria utilizzando la classe 'experience_replay' 
            for i in range(0,len(data_subset)):
                mem.remember(list(data_subset.iloc[i]))
                            
        
        #vado a campionare un batch di esperienza che sarà utilizzato per  addestrare la rete
        #print('\nGET_BATCH\n')
        batch = mem.get_batch(BATCH_SIZE) 
        print('BATCH_SIZE')
        print(len(batch))
        batch_dataset = pd.DataFrame(batch, columns = data_subset.columns)
        #calcolo la mediana del ciclo corrente, verrà utilizzata nel ciclo successivo per calcolare la reward
        median = batch_dataset['time'].median()
        print('MEDIAN JUST CALCULATED')
        print(median)
        #print(batch_dataset.iloc[:,:10])
    
        #Addestro il modello di learning e prendo il tempo di learning
        learning_start = time.time()
        agent.model_fitting(batch_dataset.drop(['test_class_name', 'cycle_id', 'time'], axis = 'columns').iloc[:, :-1], batch_dataset.drop(['test_class_name', 'cycle_id', 'time'], axis = 'columns').iloc[:,-1])
        learning_end = time.time()
        print('LEARNING TIME')
        print(learning_end - learning_start)
        learning_time.append(learning_end - learning_start)
        #setto il flag in modo tale da non rieseguire la parte relativa al primo ciclio di CI
        agent.already_fitted_model = True
   
    print('SUMMARY GENERATED')
    #print(output_data)
    

    #classification report
    #classification_report = classification_report(labels[len(labels.loc[labels['cycle_id'] == data.iloc[0]['cycle_id']][REWARD_SELECTOR]):][REWARD_SELECTOR], prediction_arry, output_dict = True)
    #print(classification_report)
    #classification_report_df = pd.DataFrame(classification_report).transpose()
    #classification_report_df.to_csv('experiments_A_time/commons_lang_summary/classification_report.csv', index = True, mode = 'a', header = True)
    #dataset di uscita
    output_data.insert(len(output_data.columns), 'prediction_time', prediction_time, allow_duplicates = True)
    output_data.insert(len(output_data.columns), 'learning_time', learning_time[1:], allow_duplicates = True)
    #output_data.to_csv('summary/' + str(args) + '-summary.csv', index = False)
    
    
    
    if not os.path.exists('history_sensitivity_analysis/4_layer'):
        os.makedirs('history_sensitivity_analysis/4_layer')
    
    if not os.path.isfile('history_sensitivity_analysis/4_layer/' + str(BATCH_SIZE) + args.replace('_result', '_injected_summary')):
        output_data.to_csv('history_sensitivity_analysis/4_layer/' + str(BATCH_SIZE) + args.replace('_result', '_injected_summary'), index = False, header = True)
    else: # else it exists so append without writing the header
        output_data.to_csv('history_sensitivity_analysis/4_layer/' + str(BATCH_SIZE) + args.replace('_result', '_injected_summary'),index = False, mode = 'a', header = False)
    
    
    
    
    #if not os.path.exists('history_sensitivity_analysis/4_layer'):
    #    os.makedirs('history_sensitivity_analysis/4_layer')
    
    #if not os.path.isfile('history_sensitivity_analysis/4_layer/' + str(BATCH_SIZE) + args.replace('_result', '_summary')):
    #    output_data.to_csv('history_sensitivity_analysis/4_layer/' + str(BATCH_SIZE) + args.replace('_result', '_summary'), index = False, header = True)
    #else: # else it exists so append without writing the header
    #    output_data.to_csv('history_sensitivity_analysis/4_layer/' + str(BATCH_SIZE) + args.replace('_result', '_summary'),index = False, mode = 'a', header = False)
    
 
 
    #for i in range(0, len(data)):
    #    x = np.array(data.iloc[i][:]).reshape(1, -1)
    #    y = pd.DataFrame(x, columns = data.columns)
    #    print(y)