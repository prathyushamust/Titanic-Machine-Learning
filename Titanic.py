import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv('/Users/pmust/Downloads/Fall_2022/AI/ML Project/train.csv')

titanic_data.head()

titanic_data.shape

(891, 12)

titanic_data.info()

titanic_data.isnull().sum()

titanic_data = titanic_data.drop(columns = 'Cabin', axis = 1)

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

print(titanic_data['Embarked'].mode())

print(titanic_data['Embarked'].mode()[0])