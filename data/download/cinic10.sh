#!/bin/bash

if [ ! -d "../cinic10/raw" ]; then
    mkdir -p ../cinic10/raw
fi

cd ../cinic10/raw

curl -o CINIC-10.tar.gz --tlsv1.2 https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz

tar -zxvf CINIC-10.tar.gz
