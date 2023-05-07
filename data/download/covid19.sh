#!/bin/bash

if [ ! -d "../covid19/raw" ]; then
    mkdir -p ../covid19/raw
fi

cd ../covid19/raw

wget https://wjdcloud.blob.core.windows.net/dataset/cycfed/covid19.tar.gz

tar -xzvf covid19.tar.gz

mv covid19/* ./

rm -rf covid19 covid19.tar.gz
