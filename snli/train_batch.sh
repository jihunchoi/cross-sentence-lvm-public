#!/usr/bin/env bash
for PARAMS_FILE in configs/$1/*.jsonnet; do
    SAVE_PATH=`basename ${PARAMS_FILE} .jsonnet`
    CMD="python -m snli.train -p ${PARAMS_FILE} -s trained/$1/${SAVE_PATH}"
    echo ${CMD}
    ${CMD}
done
