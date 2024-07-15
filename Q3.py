import numpy as np
import pandas as pd


np.random.seed(42)


years = list(range(2019, 2024))
months = list(range(1, 8))
data = []

for year in years:
    for month in months:
        
        expense = 1000 + np.random.normal(0, 100)
        data.append((year, month, expense))


df = pd.DataFrame(data, columns=["Year", "Month", "Expense"])


def random_predict(df):
    return df["Expense"].sample().values[0]


random_samples = df.sample(n=20)


random_samples["Predicted_Expense"] = random_samples.apply(lambda row: random_predict(df), axis=1)

# Calculate Mean Squared Error (MSE)
mse = np.mean((random_samples["Expense"] - random_samples["Predicted_Expense"]) ** 2)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(random_samples)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
