#!/bin/bash

input_config=$1

if [[ $input_config =~ [0-9].* ]]; then
    suffix="${BASH_REMATCH[0]}"
    algo="${suffix%.*}"
    echo "Original string: $input_config"
    echo "Suffix: $suffix"
    echo "Algo: $algo"
    (nohup python3 pipeline_runner.py $input_config > ./logs/output_$algo.log 2>&1) &
fi
