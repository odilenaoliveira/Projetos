
# Importando Bibliotecas 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

# criando função para retirada de outliers
def drop_outliers(data,col):
    iqr = 1.5 * (np.percentile(data[col], 75) - np.percentile(data[col], 25))
    data.drop(data[data[col] > (iqr + np.percentile(data[col], 75))].index, inplace=True)
    data.drop(data[data[col] < (np.percentile(data[col], 25) - iqr)].index, inplace=True)

def load_data(filename):
  dados = pd.read_csv(filename, sep=',')

  tabela = pd.DataFrame({
      'Unique':dados.nunique(),
      'Null':dados.isna().sum(),
      'NullPercent':dados.duplicated().sum(),
      'Types':dados.dtypes.values
  })
  tabela
  return dados

def transformacao(dados):
  dados.drop_duplicates(inplace=True)
  dados['AgeCar'] = 2022 - dados['Year']
  dados.drop('Year',axis=1, inplace=True)
  
  # chamando função
  drop_outliers(dados,'Price')
  drop_outliers(dados,'Mileage')
  drop_outliers(dados,'AgeCar')

  le = LabelEncoder()

  cols = dados.select_dtypes(exclude=['float']).columns
  encode = list(cols)
  dados[encode] = dados[encode].apply(lambda col: le.fit_transform(col))
  dados[encode].head()

  return dados

def selecao(dados):
  y = dados['Price']
  x = dados.drop('Price', axis=1)
  return x,y

def treinamento(x,y):
  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=42)
  scaler = MinMaxScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.fit_transform(x_test)

  return x_train, x_test, y_train, y_test

def model_regression(x_train, x_test, y_train, y_test):
  regression = {
      'FOREST':RandomForestRegressor(n_estimators=10),
      'GRADIENT':GradientBoostingRegressor(),
      'EXTRA':ExtraTreesRegressor(),
      'BAGGING':BaggingRegressor()
  }

  for nome, model in regression.items():
    print('-'*5,nome,'-'*5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Accuracy on Traing set: ", model.score(x_train,y_train))
    print("Accuracy on Testing set: ", model.score(x_test,y_test))
    print('-'*10)
    print()

def runModel(filename):
  dados = load_data(filename)
  dados = transformacao(dados)
  x,y = selecao(dados)
  x_train,x_test,y_train, y_test = treinamento(x,y)
  model_regression(x_train,x_test,y_train, y_test)


if __name__ =='__main__':
  runModel('/content/true_car_listings.csv')
