#!/bin/bash
git log --format="%H" -n 800 --reverse >log800.txt  #crea il file dei commit, 800 è il numero dei commit scelti, bisogna passare il file log800.txt allo script quando viene invocato

git checkout -b FIRST_TRY                  #crete a new local branch
git push origin FIRST_TRY                #synchronized with GitHub

while IFS='' read -r line || [[ -n "$line" ]]; do
cd  local/path/to/application_under_test #../../commons-math
    git checkout $line
    git push origin $line:FIRST_TRY --force-with-lease
    echo "$line has been pushed to the github"
    #inizia la fase di ricerca delle dipendeze della classi modificate
    start=`date +%s`
    ./script.sh #script che crea un file dependsby.txt di class test, invocando lo script dependency_searching.py (cerca le dipendenze)
    ./testing.sh dependsby.txt #script che va a testare le classi di test in dependsby.txt
    end=`date +%s`
    runtime=$((end-start))
    echo "TIME - ${runtime}"
    echo Execution time was `expr $end - $start` seconds.

    #vado a ricavare i dati relativi al testing
    python3 data_collection_new.py ${runtime} ${line}

    #dopo aver testato e collezionato i dati relativi elimino il file di testo che sarà ricreato al prossimo commit
    rm -d "dependsby.txt"
    
    
    #sleep 5
done < "$1"
