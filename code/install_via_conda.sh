# presumes:
# conda create -n deepgo python=3.6.8 anaconda

source activate deepgo
conda install -n deepgo --file=code/requirements.txt
pip install -r code/requirements_not_in_conda.txt
