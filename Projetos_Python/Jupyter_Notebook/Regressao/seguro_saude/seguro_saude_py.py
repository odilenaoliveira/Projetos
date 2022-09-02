
# Importando Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import recall_score, f1_score, precision_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def data(filename):
  df = pd.read_csv(filename,sep=',')
  return df

def transformacao(df):
  df.drop_duplicates(inplace=True)
  df =df.replace({'yes':1, 'no':0,'female':1,'male':0})
  cat = df.select_dtypes(include='O')

  df = pd.get_dummies(df, columns=[col for col in df.columns if col in cat])

  y = df.charges
  x = df.drop('charges',axis=1)
  return x,y

def treinamento(x,y):
  xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=42)

  scaler = MinMaxScaler()
  xtrain = scaler.fit_transform(xtrain)
  xtest = scaler.fit_transform(xtest)

  return xtrain, xtest, ytrain, ytest
  
# função com modelos:
def model_regression(xtrain, xtest, ytrain, ytest):
  regression = {
      'LINEAR':LinearRegression(),
      'LASSO':Lasso(alpha=1.0),
      'RIDGE':Ridge(alpha=1.0),
      'ELASTICNET':ElasticNet(alpha=0.1),
      'KNN-R':KNeighborsRegressor(n_neighbors=5),
      'DECISION TREE':DecisionTreeRegressor(max_depth=5),
      'RANDOM FOREST':RandomForestRegressor(max_depth=5),
      'ADA':AdaBoostRegressor(),
      'EXTRA':ExtraTreesRegressor(max_depth=5),
      'GRADIENT':GradientBoostingRegressor(max_depth=5)
  }
  modelos = []
  for nome, model in regression.items():

    print('->'*5,nome,'<-'*5)

    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    print('Score Train:', model.score(xtrain, ytrain))
    print('Score Test:', model.score(xtest, ytest))
    print()
    print('MAE:', mean_absolute_error(ytest, y_pred))
    print('MSE:', mean_squared_error(ytest, y_pred))
    print('RMSE:', mean_squared_error(ytest, y_pred, squared=False))
    print('MAPE:', mean_absolute_percentage_error(ytest, y_pred))
    print('->'*20)
    print()

    modelo = nome, model.score(xtest, ytest)
    modelos.append(modelo)
    tabela = pd.DataFrame(modelos, columns=['Models','Score'])
  display(tabela.sort_values('Score', ascending=False))

# excutando as funções
def runModel(filename):
  df = data(filename)
  x,y = transformacao(df)
  xtrain, xtest, ytrain, ytest = treinamento(x,y)
  model_regression(xtrain, xtest, ytrain, ytest)

# chamando as funções
if __name__ == '__main__':
  runModel('/content/insurance.csv')
