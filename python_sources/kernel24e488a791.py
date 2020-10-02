#!/bin/bash

for 0.01rate in  
do
    for epoch in {50..150..10}
    do
        python improse-layer2_metrics.py $rate $epoch > layer2'_'$rate'_'$epoch.txt
    done
done
