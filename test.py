import pandas as pd

dataset = pd.read_csv('./train.csv')
dadosTeste = pd.read_csv("./test.csv")

dataset.drop(["diferenciais"], axis=1)
dadosTeste.drop(["diferenciais"], axis=1)

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

Xteste = dadosTeste.iloc[:, 1:].values

# Tratando colunas numéricas

import numpy as np

# Tratando coluna de quartos

mean = np.mean(X[:, 3])
std = np.std(X[:, 3])

X[:, 3] -= mean
X[:, 3] /= std

mean = np.mean(Xteste[:, 3])
std = np.std(Xteste[:, 3])

Xteste[:, 3] -= mean
Xteste[:, 3] /= std

# Tratando coluna de suítes

mean = np.mean(X[:, 4])
std = np.std(X[:, 4])

X[:, 4] -= mean
X[:, 4] /= std

mean = np.mean(Xteste[:, 4])
std = np.std(Xteste[:, 4])

Xteste[:, 4] -= mean
Xteste[:, 4] /= std

# Tratando coluna de vagas

mean = np.mean(X[:, 5])
std = np.std(X[:, 5])

X[:, 5] -= mean
X[:, 5] /= std

mean = np.mean(Xteste[:, 5])
std = np.std(Xteste[:, 5])

Xteste[:, 6] -= mean
Xteste[:, 6] /= std

# Tratando coluna de área útil

mean = np.mean(X[:, 6])
std = np.std(X[:, 6])

X[:, 6] -= mean
X[:, 6] /= std

mean = np.mean(Xteste[:, 6])
std = np.std(Xteste[:, 6])

Xteste[:, 6] -= mean
Xteste[:, 6] /= std

# Tratando coluna de área extra

X[:, 7] /= 100
mean = np.mean(X[:, 7])
std = np.std(X[:, 7])

X[:, 7] -= mean
X[:, 7] /= std

Xteste[:, 7] /= 100
mean = np.mean(Xteste[:, 7])
std = np.std(Xteste[:, 7])

Xteste[:, 7] -= mean
Xteste[:, 7] /= std

# Coluna de diferenciais é redundante

X[:, 8] = 0

Xteste[:, 8] = 0

# Tratando colunas de classificação

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

Xsize = X[:, 0].size
XtesteSize = Xteste[:, 0].size

colunaTreinoTeste = labelencoder.fit_transform(np.append(X[:, 0], Xteste[:, 0]))
X[:, 0] = colunaTreinoTeste[:Xsize]
Xteste[:, 0] = colunaTreinoTeste[:XtesteSize]

Xsize = X[:, 1].size
XtesteSize = Xteste[:, 1].size

colunaTreinoTeste = labelencoder.fit_transform(np.append(X[:, 1], Xteste[:, 1]))
X[:, 1] = colunaTreinoTeste[:Xsize]
Xteste[:, 1] = colunaTreinoTeste[:XtesteSize]

Xsize = X[:, 2].size
XtesteSize = Xteste[:, 2].size

colunaTreinoTeste = labelencoder.fit_transform(np.append(X[:, 2], Xteste[:, 2]))
X[:, 2] = colunaTreinoTeste[:Xsize]
Xteste[:, 2] = colunaTreinoTeste[:XtesteSize]

""" Xsize = X[:, 8].size
XtesteSize = Xteste[:, 8].size

colunaTreinoTeste = labelencoder.fit_transform(np.append(X[:, 8], Xteste[:, 8]))
X[:, 8] = colunaTreinoTeste[:Xsize]
Xteste[:, 8] = colunaTreinoTeste[:XtesteSize] """

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2])

tabelaTreinoTeste = np.zeros((X.shape[0]+Xteste.shape[0], X.shape[1]))

for i in range(X.shape[1]):
      tabelaTreinoTeste[:, i] = np.append(X[:, i], Xteste[:, i])

X = onehotencoder.fit_transform(tabelaTreinoTeste).tocsr()[:X.shape[0], :]
Xteste = onehotencoder.fit_transform(tabelaTreinoTeste).tocsr()[X.shape[0]:, :]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.33)

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',min_samples_split=60)

regressor.fit(X_train, y_train)

y_pred_train = regressor.predict(X_train)
y_pred_test  = regressor.predict(X_test )

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho no conjunto de treinamento:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(y_train, y_pred_train) )

print('\nDesempenho no conjunto de teste:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )


Yteste = regressor.predict(Xteste)

idDados = pd.read_csv("./test.csv", header = 0)
predicao = pd.DataFrame(Yteste, columns=['preco'])
idDados = idDados[['Id']]
resposta = pd.concat([idDados, predicao], axis=1)
resposta.to_csv("resposta.csv", index=False)

""" import numpy as np

print('\nParametros do regressor:\n', 
      np.append( regressor.intercept_ , regressor.coef_  ) ) """
