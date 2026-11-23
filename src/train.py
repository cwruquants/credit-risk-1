import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.data.data_pipe import preprocess_train

def train_model(file_path: str, model_type: str = 'logistic'):
    """
    Train a logistic regression model on the preprocessed training data.

    Args:
        file_path (str): The path to the training data file.
        model_type (str): The type of model to train ('logistic' or ...).
    """
    
    # Preprocess the training data
    preprocessed_data = preprocess_train(file_path)
    
    if model_type == 'logistic':
        return train_logistic(preprocessed_data)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_logistic(preprocessed_data):
    # Define the independent variables (X) and dependent variable (y)
    X = preprocessed_data.drop(columns=['Default'])
    y = preprocessed_data['Default']

    # Add a constant to the independent variables matrix
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.Logit(y, X).fit()

    # Save the model summary to a text file
    with open('../artifacts/model_summary.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def save_model(model, file_path: str):
    """
    Save the trained model to a specified file path.

    Args:
        model: The trained model object.
        file_path (str): The path where the model will be saved.
    """
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
        
def train_store(train_file_path: str, model_file_path: str):
    """
    Main function to train the model and save it.

    Args:
        train_file_path (str): The path to the training data file.
        model_file_path (str): The path where the trained model will be saved.
    """
    model = train_model(train_file_path)
    save_model(model, model_file_path)


