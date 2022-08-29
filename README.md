# Estimating Bone Mineral Density Based on Age, Sex, and Anthropometric Measurements

Repository containing the code used for the paper `Estimating Bone Mineral Density Based on Age, Sex, and Anthropometric Measurements`, accepted for publication in BRACIS 2022. It uses nested cross-validation with Grid Search to evaluate multiple Regression Models.

## Prerequistes

- Python3
- virtualenv (or any other Python environment tool)

## Setup

- Clone the repo and change to the repo's directory
- Create a virtual environment: `virtualenv .env -p python3`
- Activate the env: `. .env/bin/activate`
- Install the requirements: `pip install -r requirements.txt`
- Create a `data` folder and move the dataset file to that folder (if you do not have the dataset file, please contact one of the authors)

## Run

- Activate the env: `. .env/bin/activate`
- Start the experiments: `python3 run_experiment.py`
- After the mains experiments are finished, run the stastiscal test: `python3 run_statistical_test.py`

## Extra info

All logs will be saved in the _logs_ folder. Outputs will be saved in the _output_ folder.

If you want to add different models or hyperparameter configurations, check the `models.py` file.
