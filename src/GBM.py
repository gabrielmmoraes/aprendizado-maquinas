import pandas as pd

dataset = pd.read_csv('../input/train.csv')
dadosTeste = pd.read_csv("../input/test.csv")

dataset = dataset.drop(columns=["diferenciais", "tipo", "tipo_vendedor"], axis=1)
dadosTeste = dadosTeste.drop(columns=["diferenciais", "tipo", "tipo_vendedor"], axis=1)

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

Xteste = dadosTeste.iloc[:, 1:].values

################################################################################################################

# Tratando colunas numéricas

import numpy as np

# Tratando coluna de quartos

mean = np.mean(X[:, 1])
std = np.std(X[:, 1])

X[:, 1] -= mean
X[:, 1] /= std

mean = np.mean(Xteste[:, 1])
std = np.std(Xteste[:, 1])

Xteste[:, 1] -= mean
Xteste[:, 1] /= std

# Tratando coluna de suítes

mean = np.mean(X[:, 2])
std = np.std(X[:, 2])

X[:, 2] -= mean
X[:, 2] /= std

mean = np.mean(Xteste[:, 2])
std = np.std(Xteste[:, 2])

Xteste[:, 2] -= mean
Xteste[:, 2] /= std

# Tratando coluna de vagas

mean = np.mean(X[:, 3])
std = np.std(X[:, 3])

X[:, 3] -= mean
X[:, 3] /= std

mean = np.mean(Xteste[:, 3])
std = np.std(Xteste[:, 3])

Xteste[:, 3] -= mean
Xteste[:, 3] /= std

# Tratando coluna de área útil

mean = np.mean(X[:, 4])
std = np.std(X[:, 4])

X[:, 4] -= mean
X[:, 4] /= std

mean = np.mean(Xteste[:, 4])
std = np.std(Xteste[:, 4])

Xteste[:, 4] -= mean
Xteste[:, 4] /= std

# Tratando coluna de área extra

mean = np.mean(X[:, 5])
std = np.std(X[:, 5])

X[:, 5] -= mean
X[:, 5] /= std

mean = np.mean(Xteste[:, 5])
std = np.std(Xteste[:, 5])

Xteste[:, 5] -= mean
Xteste[:, 5] /= std

################################################################################################################

# Tratando colunas de classificação

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

Xsize = X[:, 0].size
XtesteSize = Xteste[:, 0].size

colunaTreinoTeste = labelencoder.fit_transform(np.append(X[:, 0], Xteste[:, 0]))
X[:, 0] = colunaTreinoTeste[:Xsize]
Xteste[:, 0] = colunaTreinoTeste[:XtesteSize]

################################################################################################################

onehotencoder = OneHotEncoder(categorical_features = [0])

tabelaTreinoTeste = np.zeros((X.shape[0]+Xteste.shape[0], X.shape[1]))

for i in range(X.shape[1]):
      tabelaTreinoTeste[:, i] = np.append(X[:, i], Xteste[:, i])

X = onehotencoder.fit_transform(tabelaTreinoTeste).tocsr()[:X.shape[0], :]
Xteste = onehotencoder.fit_transform(tabelaTreinoTeste).tocsr()[X.shape[0]:, :]

################################################################################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.33)

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.01,
                                   max_depth=4, max_features=0.4,min_samples_split=300)

regressor.fit(X_train, y_train)

y_pred_train = regressor.predict(X_train)
y_pred_test  = regressor.predict(X_test )

################################################################################################################

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho no conjunto de treinamento:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(y_train, y_pred_train) )

print('\nDesempenho no conjunto de teste:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )

################################################################################################################

Yteste = regressor.predict(Xteste)

idDados = pd.read_csv("../input/test.csv", header = 0)
predicao = pd.DataFrame(Yteste, columns=['preco'])
idDados = idDados[['Id']]
resposta = pd.concat([idDados, predicao], axis=1)
resposta.to_csv("../output/resposta.csv", index=False)

""" import numpy as np

print('\nParametros do regressor:\n', 
      np.append( regressor.intercept_ , regressor.coef_  ) ) """
