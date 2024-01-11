# Project Goal

The primary objective of this project is to develop Machine Learning operations project for a deep learning-based artificial intelligence model that classifies various types of sports images. The model will be given a sports related image and it will output the name of the sports it’s related to. The purpose is to apply different number of coding practices to organizate, scale, monitor and deploy the machine learning model in a production setting.

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

In order to have a version control of the data and make the repo lighter, dvc is going to be used. To get the data, just make sure that dvc is installed in your machine. If is not, you can do it like this: 

```python
pip install dvc
```
And in this case you would also need to run the following command as we are using Google Drive to store the data.

```python
pip install "dvc[gdrive]"
```
Alternatively, installing the packages in requirements-dev.txt will also get dvc working, among other things. This is done with the following instruction:

```python
pip install -r "requirements-dev.txt"
```

## Models

The core model we expect to use is RexNet. This model is selected due to its efficiency and high accuracy in image classification tasks. We will adapt and train this model on our chosen dataset, tuning it to achieve optimal performance in sports image classification.

## Conclusion

Coming soon...

## Progress tracking

[You can find the project progress here](https://github.com/users/hmhauter/projects/1/views/1)

## Meme of the project

![Link to the meme, hope it makes you smile](https://pbs.twimg.com/media/CbzEu7eUkAAo0ag?format=jpg&name=small)

# Installation:

1. Clone the repo to your local machine.

2. Create conda environment from requirements.txt:

- Create a virtual environment:

```python
conda create -n mlops python=3.10
```

- Activate the environment using the following command:

```python
conda activate mlops
```

- Install libraries from the requirements file:

```python
pip install -r requirements.txt
```

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
