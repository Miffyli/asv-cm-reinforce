#!/bin/bash
# Concatenate scores of two directories into one 

if test -z "$1"
then
    echo "Usage: concatenate_scres.sh input_directory1 input_directory2 output_directory"
    exit
fi

for filename in ${1}/*
do
  basename=$(basename $filename)
  cat ${1}/${basename} ${2}/${basename} > ${3}/${basename}
done
