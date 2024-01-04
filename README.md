# Bilevel Optimization
## Implicit Differentiation

A project on bilevel optimization in the context of work at Thoth team INRIA under the supervision of Michael Arbel.

Make sure to add BilevelProblem and InnerSolution directories to system PATH before running.

To install dependencies:
pip install -r dependencies.txt

To run the dsprites experiment with functional bilevel framework:
python tests/Dsprites.py

To select a method of computing a*() for outer variable gradient computation:
set the variable a_star_method in line 41 in file bilevel-optimization/src/tests/Dsprites.py

To activate wandb logging:
comment line 19 in file bilevel-optimization/src/tests/Dsprites.py