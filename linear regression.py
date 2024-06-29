import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np


data = {
    'Accident_Severity': [1, 2, 3, 1, 2, 3, 2, 3, 1, 2],
    'Number_of_Vehicles': [2, 3, 1, 4, 2, 5, 3, 2, 1, 4],
    'Number_of_Casualties': [1, 2, 1, 2, 1, 3, 2, 1, 1, 2],
    'Speed_limit': [30, 40, 50, 60, 30, 70, 40, 50, 30, 60],
    'Weather_Conditions': [1, 2, 1, 3, 1, 2, 3, 1, 1, 2],
    'Road_Surface_Conditions': [1, 1, 2, 3, 1, 2, 2, 1, 1, 3]
}
df = pd.DataFrame(data)


X = df[['Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit', 'Weather_Conditions', 'Road_Surface_Conditions']]
y = df['Accident_Severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


with open('accident_severity_model.pkl', 'wb') as file:
    pickle.dump(model, file)


example_features = np.array([[3, 2, 50, 1, 2]])
predicted_severity = model.predict(example_features)
print(f'Predicted Accident Severity: {predicted_severity[0]}')


with open('accident_severity_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


predicted_severity_loaded_model = loaded_model.predict(example_features)
print(f'Predicted Accident Severity (loaded model): {predicted_severity_loaded_model[0]}')


import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Accident Severity")
plt.ylabel("Predicted Accident Severity")
plt.title("Actual vs Predicted Accident Severity")
plt.show()
