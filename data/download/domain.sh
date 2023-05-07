#!/bin/bash

if [ ! -d "../domain/raw" ]; then
    mkdir -p ../domain/raw
fi

cd ../domain/raw

wget http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip

nohup unzip clipart.zip &>/dev/null &
nohup unzip painting.zip &>/dev/null &
nohup unzip infograph.zip &>/dev/null &
nohup unzip quickdraw.zip &>/dev/null &
nohup unzip real.zip &>/dev/null &
nohup unzip sketch.zip &>/dev/null &
echo decompressing...
wait
echo Finished!
