import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

df =  pd.read_csv(r"C:\Documentos\UFSCar\DADOS2_1.csv", header=0, index_col="HORAS",na_values=1)
df = df.apply(lambda x: x.str.replace(",", ".")).astype(float)
df.shape
data = df.to_numpy().flatten()

n = data.shape[0]
input_size = 14 * 30

results = []

for i in range(n - input_size):
    x = data[i : i + input_size]
    y = data[i + input_size]

    results.append((x, y))
len(results)

x, y = results[102]

def compute_dataset(x, y):
    dx = x[-1] - x[-2]
    return x[-1], dx, y

features = [compute_dataset(x, y) for x, y in results]
features = np.array(features)

X = np.nan_to_num(features[:, :2])
Y = np.nan_to_num(features[:, -1])


reg = LinearRegression().fit(X, Y)
print(reg.coef_)

reg.intercept_

# plt.plot(reg.predict(X)[:100], label="Previu")
plt.plot((0.76808664 * X[:, 0] + 0.30797523 * X[:, 1] + 265.92722968773626)[:14*30], label="Soma (com pesos)")
plt.plot(Y[:14*30], label="Valor Real")
plt.legend()
plt.show()