#!/bin/bash

sudo apt-get install lrzip
wget 'https://www.dropbox.com/s/nkh802pdyi5bvyk/my-python3-env.tar.lrz'
lrzuntar my-python3-env.tar.lrz

sudo mkdir -p /home/ale/Programs
sudo mv my-python3-env /home/ale/Programs

source /home/ale/Programs/my-python3-env/bin/activate

#rm -rf $VIRTUAL_ENV
#sudo ln -s /home/ale/Programs/my-python3-env $VIRTUAL_ENV

mv setup.cfg setup.cfg.bak

#sudo apt-get install libblas-dev libatlas-dev liblapack-dev gfortran

#pip install -r requirements.txt --use-mirrors