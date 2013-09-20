#!/bin/bash

mv setup.cfg setup.cfg.bak

cd $HOME

sudo apt-get install lrzip
wget 'https://www.dropbox.com/s/fmp5ew0vzdd7bk1/my-python3-env.tar.lrz'
lrzuntar my-python3-env.tar.lrz

#sudo apt-get install libblas-dev libatlas-dev liblapack-dev gfortran
#pip install -r requirements.txt --use-mirrors