#!/bin/bash

wget 'https://www.dropbox.com/s/nkh802pdyi5bvyk/my-python3-env.tar.lrz'
sudo apt-get install lrzip
lrzuntar my-python3-env.tar.lrz
rm -rf $VIRTUAL_ENV
mv my-python3-env $VIRTUAL_ENV

#sudo apt-get install libblas-dev libatlas-dev liblapack-dev gfortran

#pip install -r requirements.txt --use-mirrors