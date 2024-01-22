import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('data/auto-mpg_DIG-266.csv', sep=";")

print(data)

data = data.sample(frac=1)

y_variable = data['mpg']

x_variable = data.loc[:, data.columns != 'mpg']

x_train, x_test, y_train, y_test = train_test_split(x_variable, y_variable, test_size=0.2)

regressor = LinearRegression()

regressor = regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

file_to_write = open("data/models/baummethoden_lr.pickle", "wb")

pickle.dump(regressor, file_to_write)
