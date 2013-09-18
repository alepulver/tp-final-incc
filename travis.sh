sudo apt-get install libblas-dev libatlas-dev liblapack-dev gfortran

# XXX: not recognized by scipy if not installed before
pip install 'numpy>=1.3'

pip install -r requirements.txt --use-mirrors