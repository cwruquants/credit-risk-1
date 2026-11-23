import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a given file path.

    Args:
        file_path (str): The path to the data file.
    """
    return pd.read_csv(file_path)



def preprocess_data_numerical(df: pd.DataFrame) -> pd.DataFrame:
   
   #Create the dependent variable
    df['Default'] = np.where(df[['Default Date']].isnull(), 0, 1)

    #Drop non-numeric values like dates
    num_data = df.drop(columns=['Default Date', 'Contract Date'])

    return num_data 



def preprocess_train(file_path: str):
    df = load_data(file_path)

    num_data = preprocess_data_numerical(df)

    return num_data
