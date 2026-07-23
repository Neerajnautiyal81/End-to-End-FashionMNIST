# FashionMNIST Image Classification using CNN

## Project Overview

This project is a deep learning-based image classification system that
classifies FashionMNIST images into one of 10 clothing categories using
a Convolutional Neural Network (CNN) built with PyTorch.

The application consists of: - FastAPI backend - Streamlit frontend -
PyTorch CNN model for prediction

------------------------------------------------------------------------

## Features

-   Upload FashionMNIST images
-   Predict clothing category using a CNN model
-   FastAPI REST API
-   Interactive Streamlit frontend
-   Easy to run locally

------------------------------------------------------------------------

## Tech Stack

-   Python
-   PyTorch
-   FastAPI
-   Streamlit
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib

------------------------------------------------------------------------

## Project Structure

``` text
FashionMNIST/
│
├── backend/
├── frontend/
├── model/
├── sample_images/
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone <repository-url>
cd FashionMNIST
```

Create a virtual environment:

``` bash
python -m venv venv
```

Activate it (Windows):

``` bash
venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Run the Backend

``` bash
uvicorn backend.app:app --reload
```

------------------------------------------------------------------------

## Run the Frontend

``` bash
streamlit run frontend/app.py
```

------------------------------------------------------------------------

## Model

-   Framework: PyTorch
-   Dataset: FashionMNIST
-   Architecture: CNN
-   Output Classes: 10

------------------------------------------------------------------------

## License

This project is created for educational purposes.
