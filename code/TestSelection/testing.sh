 #!/bin/bash

#questo script serve a testare le classi di test nel file .txt passato come input 
while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "#################################### {[TESTING]} ####################################"
    #echo "{[TEST CLASS]} - $line"
    mvn -Dtest=$line test -Drat.skip=true package
done < "$1"

#vado a ricavare i dati relativi al testing
#python3 data_collection.py 

#dopo aver testato e collezionato i dati relativi elimino il file di testo che sarà ricreato al prossimo commit
#rm -d "$1"
