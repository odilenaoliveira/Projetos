
# Importando bibliotecas
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, LinearSVR, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# carregando os dados
def load_data(filename):
  dados = pd.read_csv(filename, sep=',')
  return dados

# visualizando as informações dos dados
def visual_data(dados):
  print('Tamanho dos dados:', dados.shape)
  print('Dados Duplicados:', dados.duplicated().sum())
  tabela = pd.DataFrame({
      'Unique':dados.nunique(),
      'Null':dados.isna().sum(),
      'Types':dados.dtypes.values
  })
  display(tabela)
  print()
  return dados

# transformando os dados para o treinamento
def transformacao(dados):
  df = dados.copy()

  df.drop('id',axis=1, inplace=True)

  impute = SimpleImputer(missing_values=np.nan, strategy='mean')
  df['bmi'] = impute.fit_transform(df.bmi.values.reshape(-1,1))

  cat = df.select_dtypes(include='O')

  df = pd.get_dummies(df, columns = [col for col in df.columns if col in cat])

  return df

# selecionando os dados de treinamento
def selecao(df):
  y = df.stroke
  x = df.drop('stroke',axis=1)
  return x,y

# treinando os dados para os modelos
def treinamento(x,y):
  xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=42)

  # Usando o SMOTE para os dados desequilibrados
  sm = SMOTE(sampling_strategy='auto',random_state=42) 
  xtrain, ytrain = sm.fit_resample(xtrain,ytrain)
  return xtrain, xtest, ytrain, ytest
  
# Usando um loop para todos os modelos classificadores
def models_classification(xtrain, xtest, ytrain, ytest):
  # Escalonando os dados para inserir nos modelos
  scaler = MinMaxScaler()
  xtrain = scaler.fit_transform(xtrain)
  xtest = scaler.fit_transform(xtest)

  classifiers = {
      'LOGISTIC':LogisticRegression(),
      'KNN':KNeighborsClassifier(n_neighbors = 5, p = 2),
      'SVC':SVC(kernel = 'poly', degree=4),
      'FOREST': RandomForestClassifier(criterion='entropy',max_depth=4),
      'ADA':AdaBoostClassifier(),
      'DECISION': DecisionTreeClassifier(criterion='entropy', max_depth=4),
      'GRADIENT': GradientBoostingClassifier(max_depth=4)
  }
  modelos = []
  for nome, model in classifiers.items():

    model.fit(xtrain,ytrain)
    y_pred = model.predict(xtest)
    acc = nome, model.score(xtest, ytest)
    modelos.append(acc)

    tabela = pd.DataFrame(modelos, columns=['Models','Score'])


    print('->'*5,nome,'<-'*5)
    print('Train Score:', model.score(xtrain, ytrain))
    print('Test Score:', model.score(xtest, ytest))
    print()
    cm = confusion_matrix(ytest, y_pred)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(nome + ': Confusion Matrix')
    plt.show()
    print()
  display(tabela)
 
def runModel(filename):
  dados = load_data(filename)
  dados = visual_data(dados)
  df = transformacao(dados)
  x,y = selecao(df)
  xtrain, xtest, ytrain, ytest = treinamento(x,y)
  models_classification(xtrain, xtest, ytrain, ytest)

if __name__ == '__main__':
  runModel('/content/healthcare-dataset-stroke-data.csv')
