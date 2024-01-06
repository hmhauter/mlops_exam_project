#Overall Goal

The primary objective of this project is to develop a deep learning-based artificial intelligence model for classifying various types of sports images.

#Frameworks and Integration

We will utilize two frameworks for this project:

PyTorch: A well-known framework for building deep learning models.
PyTorch Lightning: An extension of PyTorch that provides a higher level of abstraction, making the training process more organized and less error-prone.
The integration of these frameworks will facilitate model development, training, and validation processes.

#Data

The initial dataset for training our model is the Fruits 100 Dataset. This dataset comprises a diverse collection of sports images, which is crucial for the effectiveness of the classification model.

#Models

The core model we expect to use is RexNet. This model is selected due to its efficiency and high accuracy in image classification tasks. We will adapt and train this model on our chosen dataset, tuning it to achieve optimal performance in sports image classification.

#Conclusion

In summary, this project aims to harness the power of deep learning to accurately classify sports images. By leveraging PyTorch and PyTorch Lightning, along with the RexNet model, we strive to develop a robust and efficient classification system. The project's success will be evaluated based on the model's accuracy and its ability to generalize across various sports images.

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
