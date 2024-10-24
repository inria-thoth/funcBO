# Functional Bilevel Optimization

A package for functional bilevel optimization, this is an implementation of our paper [Functional Bilevel Optimization for Machine Learning](https://arxiv.org/abs/2403.20233).

## Table of Contents
- [Functional Bilevel Optimization](#functional-bilevel-optimization)
  - [Table of Contents](#table-of-contents)
  - [Badges](#badges)
  - [Project Description](#project-description)
  - [Installation Instructions](#installation-instructions)
  - [Usage Examples](#usage-examples)
  - [License Information](#license-information)
  - [Support and Contact Information](#support-and-contact-information)
  - [Acknowledgements](#acknowledgements)

## Badges
![Build Status](https://img.shields.io/github/actions/workflow/status/inria-thoth/funcBO/build.yml?branch=master)
![License](https://img.shields.io/github/license/inria-thoth/funcBO)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

## Project Description
Functional Bilevel Optimization (funcBO) is a package designed to facilitate bilevel optimization problems. Bilevel optimization is a hierarchical optimization where one problem is nested within another. This package implements the methods described in our paper [Functional Bilevel Optimization](https://arxiv.org/abs/2403.20233), providing tools for both researchers and practitioners to solve complex optimization tasks with ease.

## Installation Instructions
To install dependencies:
```bash
pip install -r dependencies.txt
```
To install funcBO:
```bash
pip install -e .
```

## Usage Examples
To launch the funcID experiment on [dsprites](https://github.com/google-deepmind/dsprites-dataset) data:
```bash
python applications/IVRegression/funcBO/main.py
```
To launch the DFIV experiment on dsprites data:
```bash
python applications/IVRegression/DFIV/main.py
```
To launch an experiment or a grid search using [mlxp](https://inria-thoth.github.io/mlxp/pages/master/getting_started.html):
```bash
bash applications/IVRegression/launch_DFIV_funcBO.sh
```

## Licence Information
This project is licensed under the MIT License.

## Support and Contact Information
Please open an issue on this repository or contact the maintainers for support.

## Acknowledgements
We want to acknowledge the contributions of all the developers and researchers who have made this project possible.
