# Desaf-o-2-Grupo-5
Aquí vamos a ir dejando un historial del proyecto de properatti

# Importo librerias necesarias para esta integracion

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Importo el csv resultante del Desafio 1

df = pd.read_csv('properatti_out.csv')

# Verifico que las columnas se importaron correctamente, puedo re-usar este box para verificar operaciones ejecutadas más
# adelante

df.columns

# Instancio mi y a predecir

ys = df['PrecioM2dolar']

# Elimino columnas a no usar en x

borrar=['Ambientes', 'Precio-dolar', 'Unnamed: 0', 'ID', 'Operación', 'Tipo', 'Lugar', 'País', 'GeoID', 'Latitud', 'Longitud', 'PrecioM2dolar', 'Descripción', 'Cotizacion', 'Expensas', 'pileta']
for i in borrar:
    del df[i]

# Imputo NaN's por media

df[['SuperficietotalM2', 'SuperficiecubiertaM2']].fillna(df.mean(), inplace=True)

# Verificamos que df ya no cuenta con el valor a predecir ni variables clave que le darian demasiada ayuda

df['PrecioM2dolar', 'Precio-dolar']

# Genero dummies para Barrio y Provincia

columnas_categoricas = ['Barrio', 'Provincia']
df = pd.get_dummies(df, columns = columnas_categoricas)
df.sample(4)

# Separo mi dataframe en un set de entrenamiento y un set de verificacion

X = df
y = ys

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=53)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Genero modelos de regresión lineal, ridge y lasso, para las ultimas dos usando tecnicas de decenso de gradiente

al_ridge = np.linspace(0.001, 0.3, 300)
al_lasso = np.linspace(0.1, 0.5, 300)
kf = KFold(n_splits=5, shuffle=True, random_state=12)

lm = LinearRegression()
lm_ridge_cv= RidgeCV(alphas=al_ridge, cv=kf, normalize=False)
lm_lasso_cv = LassoCV(alphas=al_lasso, cv=kf, normalize=False)

lm.fit(X_train, y_train)
lm_ridge_cv.fit(X_train, y_train)
lm_lasso_cv.fit(X_train, y_train)

# high r2, good on training

# Imprimo los resultados de este proceso

print ("r^2 de lineal:", lm.score(X, y))
print ("r^2 de ridge:", lm_ridge_cv.score(X, y))
print ("r^2 de lasso:", lm_lasso_cv.score(X, y))
