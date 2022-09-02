
# Importando Bibliotecas
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# ML
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# Carregando Dados
def load_data(filename):
  df = pd.read_csv(filename, sep=',')

  print('Tamanho Dados:', df.shape)
  print('Dados Duplicados:', df.duplicated().sum())

  info_data = pd.DataFrame({
    'Unique':df.nunique(),
    'Null': df.isna().sum() / len(df) *100,
    'Type': df.dtypes.values
  })
  display(info_data)
  return df

# Transformando os dados
def transformacao(df):
  cat = df.select_dtypes(include='O')

  df = pd.get_dummies(df, columns=[col for col in df.columns if col in cat])
  return df

def selecao(df):

  y = df.stroke
  x = df.drop('stroke',axis=1)

  x_train, xtest, y_train, ytest = train_test_split(x,y,test_size=0.2, random_state=1)

  return x_train, xtest, y_train, ytest
  

## função para receber o treino e teste nos modelos
def models_classifier(x_train,xtest,y_train,ytest, n):
  ## pipeline 
  under = RandomUnderSampler()
  over = RandomOverSampler()

  pipeline = Pipeline(steps=[('o', over),('u',under)])

  x_train, y_train = pipeline.fit_resample(x_train,y_train)

  classifiers = {
      'SGDC':SGDClassifier(),
      'RIDGE':RidgeClassifier(),
      'RANDOM FOREST':RandomForestClassifier(n_estimators=n, criterion='entropy'),
      'DECISION TREE':DecisionTreeClassifier(),
      'GRADIENT':GradientBoostingClassifier(n_estimators=n),
      'ADA':AdaBoostClassifier(n_estimators=n),
      'EXTRA':ExtraTreesClassifier(n_estimators=n),
      'SVC':SVC(),
      'KNN':KNeighborsClassifier(n_neighbors=5)
  }

  modelos = []
  print('ESTIMATORS:',n)
  for nome, model in classifiers.items():
    print('->'*5,nome,'<-'*5)
    print()
    model.fit(x_train, y_train)
    y_pred = model.predict(xtest)
    print('Score Train:', model.score(x_train, y_train))
    print('Score Test:', model.score(xtest, ytest))
    print('Confusion Matrix:')
    print(confusion_matrix(ytest, y_pred))
    print()
    print(classification_report(ytest, y_pred))

    print()
    print('-><-'*10)
    print()

    acc = nome, model.score(xtest, ytest)
    modelos.append(acc)

    tabela = pd.DataFrame(modelos, columns=['Models','Score'])
  display(tabela.sort_values('Score',ascending=False))


def runModel(filename):
  df = load_data(filename)
  df = transformacao(df)
  x_train,xtest,y_train,ytest = selecao(df)
  models_classifier(x_train,xtest,y_train,ytest,50)
  
if __name__ == '__main__':
  runModel('/content/full_data.csv')
