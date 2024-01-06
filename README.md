# Project Goal

The primary objective of this project is to develop Machine Learning operations project for a deep learning-based artificial intelligence model that classifies various types of sports images. The model will be given a sports related image and it will output the name of the sports it’s related to. The purpose is to use different number of coding practices to organization, scale, monitor and deploy the machine learning model in a production setting that are learned throughout the course.

## Frameworks and Integration

| Framework    | Purpose / Usage |
| -------- | ------- |
| Git and GitHub  | Code Versioning    |
| TIMM | Pytorch based image models     |
| DVC    | Data Versioning and Sharing    |
| Conda    | Environment Management   |
| Python   | Coding language  |
| Pytorch   | Deep Learning freamwork  |
| VSCode and VSCode Debugger    | Code Editor and Debuger  |
| Cookiecutter   | Project template   |
| Wandb   | Experiment monitoring (and hyperparameter optimization sweeping)  |
| Ruff   | Linter, make code PEP8 compliant   |
| Docker   | Create shareable environment  |
| Hydra   | Manage Hyperparameters  |
| Pytorch-lightning   | Reduce boilerplate Code |
| More will come…  | … |

## Data

The initial dataset for training our model is the [Sports image classification](https://www.kaggle.com/datasets/sidharkal/sports-image-classification). This dataset contains 10.283 labeled images divided in two substes. The training subset contains 8227 files and the test subset contains 2056 files.

## Models

The core model we expect to use is RexNet. This model is selected due to its efficiency and high accuracy in image classification tasks. We will adapt and train this model on our chosen dataset, tuning it to achieve optimal performance in sports image classification.

## Conclusion

In summary, this project aims to harness the power of deep learning to accurately classify sports images. By leveraging PyTorch and PyTorch Lightning, along with the RexNet model, we strive to develop a robust and efficient classification system. The project's performance will be evaluated based on the model's accuracy and its ability to generalize across various sports images. We expect to use multiple parameters in order to in-depth evaluate.

## Week 1 Checklist

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages
- [x] Create the initial file structure using cookiecutter
- [ ] Fill out the make_dataset.py file such that it downloads whatever data you need and
- [ ] Add a model file and a training script and get that running
- [ ] Remember to fill out the requirements.txt file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (pep8) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] Setup version control for your data or part of your data
- [ ] Construct one or multiple docker files for your code
- [ ] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally, consider running a hyperparameter optimization sweep.
- [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

# src

Exam Project for the MLOps course at DTU January 2024.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
