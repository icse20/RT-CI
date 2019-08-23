from sklearn.datasets import load_boston 
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
import random
import time
from datetime import datetime
from sklearn import neural_network
try:
	import cPickle as pickle
except:
	import pickle


#def open_csv(filename):
#    data = []
#    with open(filename, 'r', newline='') as csvFile:
#        reader = csv.reader(csvFile, delimiter=',')
#        for row in reader:
#            data.append(row)
#    return data    

    
#La classe experience_replay è utilizzata per salvare l'esperienza. L'esperienza consiste nella tupla (stato, reward).
class experience_replay(object):
    def __init__(self, max_memory, batch_size):
	    #max_memory è l'esperienza massima che può essere memorizzata
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.memory = None

    def build_memory(self):
        memory = collections.deque(maxlen = self.max_memory)
        
        return memory
        
        
	#Memorizza l'esperienza in memory
    #L'esperienza consiste nelle tuple (stato, reward)
    def remember(self, experience):
        self.memory.append(experience)

    #Il metodo get_batch è utilizzato per ritornare un batch di esperienza randomica di dimensione batch_size.
    def get_batch(self):

        if self.batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
			#p è la probabilità di selezione di un'esperienza, la somma delle probabilità è 1
            batch_idx = np.random.choice(range(len(self.memory)), replace = False, size = self.batch_size, p = p)
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
     
        
        
class DQN_agent(object):
    def __init__(self, hidden_size, activation, warm_start, solver):
        self.hidden_size = hidden_size
        self.activation = activation
        self.warm_start = warm_start
        self.solver = solver
        
        #self.experience = experience_replay(max_memory=500)
        self.model = None
        
    def build_model(self):
        model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_size, activation=self.activation, warm_start=self.warm_start, solver=self.solver, max_iter=1200)
        
        return model
    
    def model_fitting(self, Xtrain, Ytrain):
        self.model.fit(Xtrain, Ytrain)
        
    def get_action(self, state):
        a = self.model.predict(np.array(state))
        
        return a
    
    #salva il modello nella directory corrente  
    def save_model(self, pkl_filename):
        with open(pkl_filename, 'wb') as file:  
            pickle.dump(self.model, file)
    
    #carica il modello
    def load_model(self, pkl_filename):
        with open(pkl_filename, 'rb') as file:  
            pickle_model = pickle.load(file)
        
        return pickle_model
        
##################### AGENT END #####################


def list_duplicates_remover(elems_duplicated_list):
    new_list = []
    for elem in elems_duplicated_list:
        if elem not in new_list:
            new_list.append(elem)
            
    return(new_list)


def xml_searching(path, test_class_name):
    
    for root, dirs, files in os.walk(path):
        #print(len(files))
        for x in files:
            #print(x)
            if x.endswith("." + test_class_name.replace("java","xml")):    #trasforma xTest.java in .xTest.xml, in questo modo prendo solo il file xml giusto
                return(x)                                                 #evitando file che hanno nomi che includono xTest.java (yxTest.java)


def features_init(test_classes):
    features_data = pd.DataFrame()      #empty dataframe
    for test_class in test_classes:
        #discutere sui valori di inizializzazione delle features, numero di verdicts e rispettivi valori
        temp_dataframe = pd.DataFrame({"Id":[str(test_class)], "time":[0.0], "time_since":[0.0], "CountLineBlank":[0.0], "CountLineCode":[0.0],  "failures_%_1":[0.0], "failures_%_2":[0.0], "failures_%_3":[0.0], "failures_%_4":[0.0], "verdicts_1":[0.0], "verdicts_2":[0.0], "verdicts_3":[0.0], "verdicts_4":[0.0]})
        features_data = features_data.append(temp_dataframe, ignore_index = True)

    return(features_data)


def features_builder(data, test_classes):
    #vado a creare lo stato, ovvero per ogni classe di test modificata vado a recuperne le features che saranno utilizzate per la predizione

    print(data)
    #seleziono solo le classi di test che mi interessano (quelle modificate) nei dati storici delle classi di test, EVENTUALI DATI STORICI NON REPERIBILI
    features_data = data.loc[data['Id'].isin(test_classes)] 
    
    #calcolo il tempo dall'ultima esecuzione della generica classe di test e la differenza tra metriche
    metric_list = ['CountLineBlank', 'CountLineCode']
    db = understand.open("/Users/romolodevito/Desktop/commons-lang/commons-lang.udb")  #apro il database understand
    for i in range(0,len(features_data)):
        features_data.iloc[i, features_data.columns.get_loc('time_since')] = round((time.time() - features_data.iloc[i, features_data.columns.get_loc('time_since')])/60) #in minuti
        entity = db.lookup(features_data.iloc[i]['Id'], "file")
        print(entity)
        for ent in entity:
            if ent.name() == features_data.iloc[i]['Id']:
                print(ent.name())
                ent_metrics = ent.metric(metric_list) #metrics is a dictionary  (ent.metrics())
                for key, value in ent_metrics.items():
                    if key in features_data.columns:
                        features_data.iloc[i, features_data.columns.get_loc(key)] = value - features_data.iloc[i, features_data.columns.get_loc(key)]
            
            #print(ent.longname().split('/')[-1])
                print(ent_metrics)
    db.close()
       
    print('FEATURES_DATA')    
    print(features_data)   
   
    #serve a creare le features delle classi di test non presenti nei dati storici, 
    #ovvero quelle classi di test che non sono state mai eseguite e di conseguenza non si hanno dati storici a disposizione
    print('---CLASSI NON NEI DATI STORICI---')
    for test_class in test_classes:
        if test_class not in list(data.iloc[:, 0]):
            #discutere sui valori di inizializzazione delle features, numero di verdicts e rispettivi valori
            temp_dataframe = pd.DataFrame({"Id":[str(test_class)], "time":[0.0], "time_since":[0.0], "CountLineBlank":[0.0], "CountLineCode":[0.0], "failures_%_1":[0.0], "failures_%_2":[0.0], "failures_%_3":[0.0], "failures_%_4":[0.0], "verdicts_1":[0.0], "verdicts_2":[0.0], "verdicts_3":[0.0], "verdicts_4":[0.0]})
            features_data = features_data.append(temp_dataframe, ignore_index = True)
            print(test_class)
    print('---CLASSI NON NEI DATI STORICI---')
    print(features_data)
    
    return(features_data)


#passare anche il numero di verdetti che si vuole per ogni riga
def data_updater(features_data):
    #dopo aver eseguito le classi di test vado a recuperare i dati risultanti dall'esecuzione, quali: numero di tests per classe, fallimenti, tempo di esecuzione
    for i in range(0,len(features_data)):
        #costruisco un buffer circolare con i verdetti, in questo modo quando inserirò il valore di verdetto più recente, in seguito all'esecuzione dei test,
        #il valore più vecchio sarà automaticamente eliminato dal buffer
        verdicts_ring = collections.deque(maxlen=4)
        failures_ring = collections.deque(maxlen=4)
        for j in range(0, len(features_data.iloc[i][-4:])):   #le ultime posizioni contegono i verdetti (nel caso specifico le ultime 4)
            verdicts_ring.append(features_data.iloc[i][-4+j])
            failures_ring.append(features_data.iloc[i][-8+j])
        
        #cerco l'xml con i dati dell'esecuzione e aggiorno i dati storici in mio possesso
        xml_name = xml_searching("target/surefire-reports", features_data.iloc[i]['Id']) #nella posizione 0 dell'header c'è l'Id, ovvero il nome della classe di test
        if xml_name:
            #parsing del file xml
            tree = ET.parse("target/surefire-reports/" + xml_name)  
            root = tree.getroot()
            verdicts_ring.append(float(i+1))#verdicts_ring.append('appended') #ELIMINARE POI
            failures_ring.append(float(i+1))#verdicts_ring.append('appended') #ELIMINARE POI
            #inserisco 1 se almeno un test è fallito (failures > 0), 0 altrimenti, potrei anche inserire la percentuale di test falliti (test_falliti/test_totali)
            verdicts_ring.append(1.0 if float(root.attrib.get('failures')) > 0 else 0.0)
            failures_ring.append(float(root.attrib.get('failures')) / float(root.attrib.get('tests')))
            #print(root.attrib.get('name').split('.')[-1] + '.java', float(root.attrib.get('tests')), float(root.attrib.get('failures')), float(root.attrib.get('time')))
            features_data.iloc[i, features_data.columns.get_loc('time')] = float(root.attrib.get('time'))
            #aggiorno i valori dei verdetti nei dati storici delle classi di test
            for j in range(0, len(features_data.iloc[i][-4:])):
                features_data.iloc[i, features_data.columns.get_loc('verdicts_' + str(j+1))] = verdicts_ring[j]  
                features_data.iloc[i, features_data.columns.get_loc('failures_%_' + str(j+1))] = failures_ring[j]
        
        print('verdicts_ring')
        print(verdicts_ring)
        print('failures_ring')
        print(failures_ring)
    
    return(features_data)



#funzione che attribuisce le reward
def reward(features_data):
        print('REWARD')
        features_data.insert(len(features_data.columns),'reward', features_data.sum(axis=1), allow_duplicates = True)
        print(features_data.sum(axis=1))
        return(features_data)

if __name__ == '__main__':

    #with open('historical_data.csv', 'w+') as outcsv:
    #    writer = csv.writer(outcsv)
    #    writer.writerow(['Id', 'durata', 'time_since', 'verdicts_1', 'verdicts_2', 'verdicts_3', 'verdicts_4'])
    

    #creo dataFrame da un array  
    #labels = ['Id', 'durata', 'time_since', 'verdicts_1', 'verdicts_2', 'verdicts_3', 'verdicts_4' ]
    #pd_dataset = pd.DataFrame.from_records(dataset, columns=labels)
    #print(pd_dataset.drop('Id', axis = 'columns'))    
        
                
    print('################################################################################################################################################################')
    print('\n DEFINITIVO \n')

    #leggo le testsuite da testare dal file dependsby.txt
    in_file = open("dependsby.txt","r")
    duplicated_test_classes = in_file.read().split(",")[:-1]  
    test_classes =  list_duplicates_remover(duplicated_test_classes)
    in_file.close()
    
    #istanziazione agente
    agent = DQN_agent((32,16), 'relu', False, 'adam')
        
    try:
        
        #carico il modello di learning
        agent.model = agent.load_model("pickle_model.pkl")
        print('\nMODEL_EXIST\n')
        
        #carico dati storici ed esperienza
        #DOMANDA = come trattiamo i casi di test nuovi
        #vado a creare lo stato, ovvero per ogni classe di test modificata vado a recuperne le features che saranno utilizzate per la predizione
        data = pd.read_csv('features_data.csv', header = 0)
        features_data = features_builder(data, test_classes)

        #Predico la reward e ordino le classi di test in base alla stessa
        action = agent.get_action(features_data.drop('Id', axis = 'columns'))
        #aggiungo la colonna della reward ai dati
        features_data.insert(len(features_data.columns),'reward', action, allow_duplicates = True)     #len(new_features_data.columns) ritorna il numero di colonne
        #ordino le classi di test secondo la reward predetta dal modello di learning
        features_data = features_data.sort_values(by = 'reward', ascending = False)
        
        print('LOAD MODEL - SORTED_FEATURES_DATA')
        print(features_data)
        
        
        
    except:
        
        print('\nMODEL_INIT\n')
        
        #inizializzo il modello di learning
        agent.model = agent.build_model()
        
        #inizializzo le features delle classi di test, compreso la reward (attribuita randomicamente nel primo passo di CI)
        #features_data = pd.DataFrame()      #empty dataframe
        #for test_class in test_classes:
        #    #discutere sui valori di inizializzazione delle features, numero di verdicts e rispettivi valori
        #    temp_dataframe = pd.DataFrame({"Id":[str(test_class)], "time":[0.0], "time_since":[0.0], "CountLineBlank":[0.0], "CountLineCode":[0.0],  "failures_%_1":[0.0], "failures_%_2":[0.0], "failures_%_3":[0.0], "failures_%_4":[0.0], "verdicts_1":[0.0], "verdicts_2":[0.0], "verdicts_3":[0.0], "verdicts_4":[0.0]})
        #    features_data = features_data.append(temp_dataframe, ignore_index = True)
            #print(test_class)
       
        #inizializzo le features delle classi di test, compreso la reward (attribuita randomicamente nel primo passo di CI)
        features_data = features_init(test_classes)
       
        #inserisco nel dataset la colonna ralativa alla 'reward', generata ramdomicamente (posso farlo anche direttamente in fase di generazione dei dati)
        action = [np.random.randint(0,50) for i in range(0,len(features_data))]
        #aggiungo la colonna della reward ai dati
        features_data.insert(len(features_data.columns),'reward', action, allow_duplicates = True)     #len(new_features_data.columns) ritorna il numero di colonne
        
        print(features_data)
        
        #ordino le classi di test secondo la reward generata randomicamente in precedenza
        features_data = features_data.sort_values(by = 'reward', ascending = False)
        
        print('INIT MODEL - SORTED_FEATURES_DATA')
        print(features_data)
        
        
    #Creo file.txt con classi di test ordinate
    #scrivo sul file solo il nome delle classi di test da testare in ordine di priorità 
    #non lascio spazi tra una classe e l'altra in modo tale da eseguire le testsuite in una sola build mvn   
    file = open("prioritized_test_classes.txt","a+") 
    #line rappresenta una riga del dataset 'features_data', prendo solo il campo relativo alla prima colonna (nome della classe di test) 
    for line in features_data.iloc[:,0]:   
        print(line)     
        file.write("%s," % line)      
    file.close()    
    
    #Eseguo classi di test secondo l'ordine chiamando lo script 'testing_script.sh' che chiama il comando mvn test
    #subprocess.check_call(['./testing_script.sh', 'prioritized_test_classes.txt'])
    
    #Aggiorno i dati storici con i dati dell'esecuzione e li salvo
    features_data = data_updater(features_data.drop('reward', axis = 'columns'))
    print(features_data)
    
    
    #try:
        #se ho dati storici non sono al primo ciclo di CI, aggiorno i dati storici con i dati relativi all'esecuzione e li salvo in un csv
    #    updated_data = pd.concat([data, features_data]).drop_duplicates(['Id'], keep = 'last')
    #    updated_data.to_csv('historical_data.csv', index = False)
    #    print(pd.concat([data, features_data]).drop_duplicates(['Id'],keep='last'))#.sort_values('Code')
        
    #except:
        #se non ho dati storici sono al primo ciclo di CI e quindi salvo solo i dati generati nel corrente ciclo
    #    features_data.to_csv('features_data.csv', index = False)
    
  
    #Attribuisco una reward alle classi di test
    features_data = reward(features_data)
    print(features_data)
    
    
    #istanzio memoria manager
    mem = experience_replay(500,5)   #max_memory, batch_size
    
    try:
        #carico l'esperienza
        mem.memory = mem.load_memory('experience.pkl')
        print('\nMEMORY_EXIST\n')
        #print(mem.memory)
        
        
    except:
        print('\nMEMORY_INIT\n')
        #inizializzo la memoria
        mem.memory = mem.build_memory()
        
    
    #aggiorno l'esperienza (state, reward) e la salvo 
    metric_list = ['CountLineBlank', 'CountLineCode']
    db = understand.open("/Users/romolodevito/Desktop/commons-lang/commons-lang.udb")  #apro il database understand
    for i in range(0, len(features_data)):
        mem.remember(list(features_data.iloc[i]))
        features_data.iloc[i, features_data.columns.get_loc('time_since')] = time.time()
        entity = db.lookup('.' + features_data.iloc[i]['Id'], "file")
        for ent in entity:
            if ent.name() == features_data.iloc[i]['Id']:
                #print(ent.name())
                ent_metrics = ent.metric(metric_list) #metrics is a dictionary  (ent.metrics())
                for key, value in ent_metrics.items():
                    if key in features_data.columns:
                        features_data.iloc[i, features_data.columns.get_loc(key)] = value
    
    db.close()
    print(mem.memory)
    
    #mem.save_memory('experience.pkl')

    
    #Effettuo sampling randomico dell'esperienza
    print('\nGET_BATCH\n')
    batch = mem.get_batch()
    print(batch)    
    
    #creo un dataframe pandas a partire dall'esperienza campionata precedentemente
    labels = ['Id', 'durata', 'time_since', 'CountLineBlank',  'CountLineCode', 'failures_%_1', 'failures_%_2', 'failures_%_3', 'failures_%_4', 'verdicts_1', 'verdicts_2', 'verdicts_3', 'verdicts_4', 'reward' ]
    batch = pd.DataFrame.from_records(batch, columns = labels)
    print(batch)
    
    #Addestro il modello di learning
    agent.model_fitting(batch.drop('Id', axis = 'columns').iloc[:, :-1], batch.drop('Id', axis = 'columns').iloc[:,-1])
    #print(batch.drop('Id', axis = 'columns').iloc[:,:-1])
    #print(batch.drop('Id', axis = 'columns').iloc[:,-1])
    
    #Salvo dati storici, esperienza e modello di learning
    try:
        #se ho dati storici non sono al primo ciclo di CI, aggiorno i dati storici con i dati relativi all'esecuzione e li salvo in un csv
        updated_data = pd.concat([data, features_data.drop('reward', axis = 'columns')]).drop_duplicates(['Id'], keep = 'last')
        updated_data.to_csv('features_data.csv', index = False)
        print('DATA_concat')
        print(pd.concat([data, features_data.drop('reward', axis = 'columns')]).drop_duplicates(['Id'],keep='last'))#.sort_values('Code')
        
    except:
        #se non ho dati storici sono al primo ciclo di CI e quindi salvo solo i dati generati nel corrente ciclo
        features_data.drop('reward', axis = 'columns').to_csv('features_data.csv', index = False)
    
    
    mem.save_memory('experience.pkl')
    agent.save_model("pickle_model.pkl")
    
    
###########    
    
    #creare dataset con label delle colonne a partire da lista
    #labels = ['Id', 'durata', 'time_since', 'verdicts_1', 'verdicts_2', 'verdicts_3', 'verdicts_4', 'reward' ]
    #pd_dataset = pd.DataFrame.from_records(batch, columns=labels)
    #print(pd_dataset)
     
     
    #fmt = '%Y-%m-%d %H:%M:%S'
    #tstamp1 = datetime.strptime('2016-04-06 21:26:27', fmt)
    #tstamp2 = datetime.now()
    #td = tstamp2 - tstamp1
    #td_mins = int(round(td.total_seconds() / 60))
    #print(round(td.total_seconds()/60))
    #print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
       
       
    #db = understand.open("/Users/romolodevito/Desktop/commons-lang/commons-lang.udb")  #apro il database understand
     
    #metrics = db.metric(db.metrics()) #metrics is a dictionary  
    #for k,v in sorted(metrics.items()):
    #    print (k,"=",v)
     
    #entity = db.lookup("EqualsBuilder.java", "file") #"EqualsBuilder.java" ritorna oggeto ENT
    #print(entity)
    
    #for ent in entity:
    #    ent_metrics = ent.metric(ent.metrics()) #metrics is a dictionary  
    #    for key,value in sorted(ent_metrics.items()):
    #        print (key,"=",value)
    
    #db.close()