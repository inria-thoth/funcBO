## Functional Bilevel Optimization

A package for functional bilevel optimization, this is an implementation of our paper https://arxiv.org/abs/2403.20233.

To install dependencies:
pip install -r dependencies.txt

To install funcBO:
pip install -e .

To launch the funcID experiment on dsprites data:
python applications/IVRegression/funcBO/main.py

To launch the DFIV experiment on dsprites data:
python applications/IVRegression/DFIV/main.py

To launch an experiment or a grid search using mlxp (https://pypi.org/project/MLXP/):
bash applications/IVRegression/launch_DFIV_funcBO.sh
