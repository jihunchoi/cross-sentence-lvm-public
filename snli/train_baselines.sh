#!/usr/bin/env bash
for PARAMS_FILE in configs/snli/baseline/*.jsonnet; do
    SAVE_PATH=`basename ${PARAMS_FILE} .jsonnet`
    CMD="python -m snli.baseline.train -p ${PARAMS_FILE} -s trained/baseline/${SAVE_PATH}"
    echo ${CMD}
    ${CMD}
done
