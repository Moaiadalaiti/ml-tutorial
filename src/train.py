import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv(r'data/auto-mpg.csv')
print(df.columns)
x = np.array(df['gewicht']).values#.reshape(-1,1)
y = np.array(df['mpg']).values#.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, testsize=0.2, random_state=0.42)
model = LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
accuracy_score(y_test, predictions)

pickle.dump(model, open('/src/train.py', 'wb'))


