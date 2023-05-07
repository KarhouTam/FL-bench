#!/bin/bash

if [ ! -d "../medmnistC/raw" ]; then
    mkdir -p ../medmnistC/raw
fi

cd ../medmnistC/raw

wget https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistC.tar.gz

tar -xzvf medmnistC.tar.gz

mv medmnistC/* ./

rm -rf medmnistC
