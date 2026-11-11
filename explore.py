import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/UCI_Credit_Card.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)