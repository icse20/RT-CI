#!/bin/bash

for dataset in codec compress imaging io lang math
do
    ./repetitions.sh ${dataset}
    echo "Completed ${dataset}"
done