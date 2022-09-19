# -*- coding: utf-8 -*-
"""fumante_nao_fumante.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ToKyDLqrEj5rEcSHvWBRXisgAzFX1ms8
"""



# Commented out IPython magic to ensure Python compatibility.
# Importando os dados
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')
# %matplotlib inline

# Carregando os dados
def load_data(filename):
  dados = pd.read_csv(filename)

  ## Informação dos dados
  print('Tamanho dos dados:',dados.shape)
  print('Dados Duplicados:', dados.duplicated().sum())

  info_data = pd.DataFrame({
      'Unique':dados.nunique(),
      'Null':dados.isna().sum(),
      'NullPercent':round(dados.isna().sum() / len(dados)*100),
      'Type':dados.dtypes.values
  })
  info_data

  return dados

# Transformando os dados
def transform_data(dados):
  #dados = dados[dados['gender'] != 'Other']

  df = dados.copy()
  df = df[df['gender'] != 'Other']
  
  impute = SimpleImputer(missing_values=np.nan, strategy='mean')
  df['bmi'] = impute.fit_transform(df.bmi.values.reshape(-1,1))

  #dados['bmi'] = impute.fit_transform(dados.bmi.values.reshape(-1,1))

  cat = df.select_dtypes(include='O')

  ## One Hot Encoding
  df = pd.get_dummies(df, columns = [col for col in df.columns if col in cat])
  return df


# Separando os dados
def split_data(df):
  y = df.stroke
  x = df.drop(['stroke'], axis=1).values
  x_columns = df.drop(['stroke'], axis=1)

  ## dados desbalanceados
  sm = SMOTE(sampling_strategy='auto',random_state=42) 
  x, y = sm.fit_resample(x, y)
  return x,y


# Treinando os dados
def train_data(x,y, testSize, num):
  # separando os daods para treino e teste
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=testSize, random_state=num)
  return x_train, x_test, y_train, y_test


# Modelo de classificação
def model_train(x_train, x_test, y_train, y_test):
  model = GradientBoostingClassifier(max_depth=8, n_estimators=300, max_features=3, random_state=42)
  model.fit(x_train,y_train)

  y_pred = model.predict(x_train)
  y_pred_t = model.predict(x_test)

  print('Train Score:',model.score(x_train, y_train))
  print('Test Score:',model.score(x_test, y_test))

  print('Classification Report')
  print(classification_report(y_test, model.predict(x_test)))
  return model, y_test, x_test


# Visualização da Predição
def visual_data(model ,y_test, x_test):
  # Matrix de Confusão
  cm = confusion_matrix(y_test, model.predict(x_test))

  # Visualizando a porcentagem através da matrix
  cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

  # plotando a predição
  fig,axes = plt.subplots(1,2,figsize=(14,7))
  sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
  axes[0].set_title('Valor da predição')

  sns.heatmap(cm_norm, annot=True, ax=axes[1], cmap='Blues')
  axes[1].set_title('Porcentagem do valor de predição')
  return plt.show()


# Chamando a função
def runModel(filename):
  dados = load_data(filename)
  df = transform_data(dados)
  x,y = split_data(df)
  x_train, x_test, y_train, y_test = train_data(x,y, 0.3, 42)
  model, y_test, x_test = model_train(x_train, x_test, y_train, y_test)
  visual_data(model, y_test, x_test)

# Executando a função
if __name__ == '__main__':
  runModel('/content/healthcare-dataset-stroke-data.csv')