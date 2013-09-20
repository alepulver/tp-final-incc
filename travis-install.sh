#!/bin/bash

sudo apt-get install lrzip
wget 'https://www.dropbox.com/s/fmp5ew0vzdd7bk1/my-python3-env.tar.lrz'
lrzuntar my-python3-env.tar.lrz

mv setup.cfg setup.cfg.bak

#sudo apt-get install libblas-dev libatlas-dev liblapack-dev gfortran
#pip install -r requirements.txt --use-mirrors