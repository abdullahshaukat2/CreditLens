import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/UCI_Credit_Card.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)

#missing values in the dataset
print("Missing values:", df.isna().sum().sum())

#default rate

default_rate = df["default.payment.next.month"].mean()
print("Default Rate:", default_rate)

#last month payment status distribution vs default rate

#count
print(df['PAY_0'].value_counts().sort_index())
#default rate
print(df.groupby('PAY_0')['default.payment.next.month'].mean())
plt.figure(figsize=(10, 5))
df.groupby('PAY_0')['default.payment.next.month'].mean().plot(kind='bar')
plt.xlabel('Payment Status')
plt.ylabel('Default Rate')
plt.title('Default Rate by Payment Status')
plt.show()

print(df["EDUCATION"].value_counts().sort_index())
print(df["MARRIAGE"].value_counts().sort_index())




# 1. What % defaulted, and why is accuracy a bad metric here?
#
# 2. How does default rate change as PAY_0 goes up?
#
# 3. Should a real lender use SEX, AGE, MARRIAGE? Why or why not?
#


