import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
# Pichle 5 dino ka data (Dummy Data)
data = {
    'day': [1, 2, 3, 4, 5],
    'orders': [40, 45, 52, 58, 70]
}

df = pd.DataFrame(data)

# Machine Learning Model
X = df[['day']] 
y = df['orders']

model = LinearRegression()
model.fit(X, y)

# Agle din (Day 6) ki prediction
prediction = model.predict(pd.DataFrame([[6]], columns=['day']))
plt.scatter(df['day'], df['orders'], color='blue')
plt.plot(df['day'], model.predict(X), color='red')
plt.scatter([6], prediction, color='green', marker='*', s=200, label='Day 6 Prediction')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Food Orders')
plt.title('Food Order Prediction System')
plt.grid(True) # Isse graph mein piche lines (jali) ban jayengi
plt.savefig('my_prediction_graph.png')
print("Day 6 ke liye predicted orders:", prediction[0])
print("-" * 30)
print(f"Agle din ke liye Predicted Orders: {int(prediction[0])}")
print("-" * 30)