import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data
data = pd.DataFrame({
    'feature1': [10, 20, 30, 40, 50],
    'feature2': [100, 200, 300, 400, 500],
    'price': [15, 25, 35, 45, 55]
})

X = data[['feature1', 'feature2']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
