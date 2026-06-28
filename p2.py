from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
    "years": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "salary": [1200, 1500, 1800, 2200, 2600, 3000, 3400, 3800, 4200, 4600]
})


X = data[["years"]]
y = data["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 42,
    test_size = 0.2
)

model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("R2 = ", r2_score(y_test, y_pred))

print("MSE = ", mean_squared_error(y_test, y_pred))

print(model.predict([[95]]))