import understand
import sys

#cambiare path e nome del database

# funzione che cerca le dipendenze e crea un file con le classi test
def dependency_searching(entities, lookup, file, depth):
    not_test_array = []
    
    #se la classe è una classe di test la inserisco direttamente nella lista dei file da testare, se non lo è vado a cercare le dipendenze
    for ent in entities:
        if lookup in ent.name():
            file.write("%s," % ent)
            #file.write("%s\r\n" % ent)
        else:
            refs = ent.dependsby() #ritorna un dizionario [(ent,list of refs)]
            #print(refs.keys())
            for key in refs.keys():
                if lookup in key.name(): #name è un metodo della classe ENT di Understand
                    print("IN - " + key.name())
                    file.write("%s," % key)
                    #file.write("%s\r\n" % key)       
                else:
                    #print("NOT IN - " + key.name())
                    not_test_array.append(key)
    
    if (not_test_array and depth > 0):
        depth = depth - 1
        dependency_searching(not_test_array, lookup, file, depth)


if __name__ == '__main__':
    
    #Open Database
    args = sys.argv[1]   #passo un parametro in più a python3. Oltre allo script da eseguire passo la classe per cui andrò a trovare le dipendenze
    db = understand.open("/Users/romolodevito/Desktop/commons-math/commons-math.udb")  #apro il database understand
  
    #file su cui scrivo classi di test da testare tramite mvn  
    file = open("dependsby.txt","a+") 

    #cerco classi modificate/aggiunte nel database Understand e ritorno oggeto ENT
    entity = db.lookup(args, "file") #"EqualsBuilder.java" ritorna oggeto ENT
    print(entity)        
            
    #funzione che trova le dipendenze e genera un file.txt contenente le classi di test da testare 
    #tale funzione permette di impostare il livello di profondità delle dipendenze (in questo caso depth è 1)  
    dependency_searching(entity, "Test", file, 1)

    #chiudo file di testo e database understand
    file.close()
    db.close()

        
