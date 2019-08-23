#!/bin/bash

for dataset in compress imaging io lang math
do
    ./sensitivity.sh ${dataset}
    echo "Completed ${dataset}"
done