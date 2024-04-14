import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data = pd.read_csv("data/raw/imdb_movies.csv")

duplicate=data[data.duplicated("orig_title")]

data.drop_duplicates(subset="orig_title",inplace=True)

data["date_x"]=data["date_x"].str.strip()
data["date_x"]=pd.to_datetime(data["date_x"],format="%m/%d/%Y")
data["Release_year"]=data["date_x"].dt.year
data["Release_month"]=data["date_x"].dt.month

data=data.dropna()

recom_data=data.copy()

folder_path = "data/processed"

file_name = "recom_data_processed.csv"

file_path = f"{folder_path}/{file_name}"

recom_data.to_csv(file_path, index=False)

print(f"The preprocessed data has been saved at: {file_path}")