from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

data = {
    "years": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [35000, 40000, 50000, 60000, 65000, 70000, 80000, 90000, 95000, 105000]
}

df = pd.DataFrame(data)

X = df[["years"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 42,
    test_size = 0.2
)

model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print(r2_score(y_test, y_pred))

print(mean_squared_error(y_test, y_pred))

print(model.predict([[7.5]]))