#!/bin/bash

if [ ! -d "../cinic10/raw" ]; then
    mkdir -p ../cinic10/raw
fi

cd ../cinic10/raw

wget https://datashare.ed.ac.uk/download/DS_10283_3192.zip

unzip DS_10283_3192.zip

tar -zxvf CINIC-10.tar.gz
