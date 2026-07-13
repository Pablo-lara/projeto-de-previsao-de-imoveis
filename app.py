import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

df = pd.read_csv("data/housing.csv")

print("="*50)
print("Primeiras linhas")
print(df.head())

print("="*50)
print("Informações do Dataset")
print(df.info())

print("="*50)
print("Estatísticas")
print(df.describe())

print("="*50)
print("Valores nulos")
print(df.isnull().sum())

# Remove duplicados
df.drop_duplicates(inplace=True)

# Preenche valores nulos (caso existam)
df.fillna(df.median(numeric_only=True), inplace=True)

print("="*50)
print("Dataset após limpeza")
print(df.info())


df = df.drop(columns=["ocean_proximity"])
# Coluna alvo (preço do imóvel)
y = df["median_house_value"]

# Variáveis independentes
X = df.drop(columns=["median_house_value"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("="*50)
print("Treino:", X_train.shape)
print("Teste :", X_test.shape)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

print("="*50)
print("Modelo treinado com sucesso!")

predicoes = modelo.predict(X_test)

print("="*50)
print("Primeiras previsões:")

for real, previsto in zip(y_test.head(), predicoes[:5]):
    print(f"Real: {real:.2f} | Previsto: {previsto:.2f}")

mae = mean_absolute_error(y_test, predicoes)

rmse = np.sqrt(mean_squared_error(y_test, predicoes))

r2 = r2_score(y_test, predicoes)

print("="*50)
print("MÉTRICAS")

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

joblib.dump(modelo, "modelo_previsao_imoveis.pkl")

print("="*50)
print("Modelo salvo com sucesso!")

modelo_salvo = joblib.load("modelo_previsao_imoveis.pkl")

novo_imovel = pd.DataFrame({
    "longitude": [-118.40],
    "latitude": [34.25],
    "housing_median_age": [20],
    "total_rooms": [4500],
    "total_bedrooms": [900],
    "population": [1200],
    "households": [850],
    "median_income": [8.5]
})

preco = modelo_salvo.predict(novo_imovel)

print("="*50)
print(f"Preço previsto: US$ {preco[0]*100000:,.2f}")

