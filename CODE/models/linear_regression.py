from sklearn.linear_model import LinearRegression


lr = LinearRegression()
model = lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

# print(y_pred)
# print(y_train)
r_sq = model.score(X_val, y_val)
print('coefficient of determination:', r_sq)