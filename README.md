## Functional Bilevel Optimization

A project on bilevel optimization in the context of work at Thoth team INRIA under the supervision of Michael Arbel and Julien Mairal.

To install dependencies:
pip install -r dependencies.txt

To install funcBO:
pip install -e .

To launch funcBO:
python applications/IVRegression/funcBO/main.py

To launch DFIV:
python applications/IVRegression/DFIV/main.py

To launch an experiment:
bash applications/IVRegression/launch_DFIV_funcBO.sh