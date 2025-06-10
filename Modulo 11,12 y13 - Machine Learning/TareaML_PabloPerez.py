# Cargo las librerias 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold




# Establecemos nuestro escritorio de trabajo

os.chdir('C:/Users/pablo/Master Big Data/Modulo 11 - Machine Learning (Inmaculada)/Tarea')

data = pd.read_excel('datos_tarea25.xlsx')

# Eliminacion de duplicados
data.duplicated().sum()
data = data.drop_duplicates().reset_index(drop = True)

data.head()

# Comprobación de los tipos de las variables
data.dtypes

data.groupby('Cylinders').size()
# Correción de errores en el tipo

data['Levy'] = pd.to_numeric(data['Levy'], errors='coerce')  # Convierte y pone NaN donde no pueda

# Creación de una nueva variable para determinar si tiene turbo o no
data[['Engine volume', 'Engine type']] = data['Engine volume'].str.split(' ', n=1, expand=True)
data['Engine volume'] = data['Engine volume'].astype(float)
data['Engine type'] = data['Engine type'].fillna('No Turbo')

data['Mileage']=data['Mileage'].str.replace('km',' ').astype('int')

data['Cylinders']=data['Cylinders'].astype('object')

# Comprobamos que se han modificado los tipos correctamente
data.dtypes



variables = list(data.columns) 

key_cols = ['Price', 'Levy', 'Manufacturer', 'Prod.year', 'Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage', 'Cylinders'
            , 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Airbags', 'Engine type']

# Seleccionar las columnas numéricas del DataFrame
numericas = data.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleccionar las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]
 

# Analisis descriptivo de variables numericas
descriptivos_cuantitativas=data.describe().T


# Grafico para identificar outliers
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))  
axes = axes.flatten()  

# Graficamos los boxplot de cada variable numerica
for i, col in enumerate(numericas):
    sns.boxplot(data=data, x=col, ax=axes[i])
    axes[i].set_title(f'Boxplot de "{col}"')

plt.tight_layout()
plt.show()



for column in numericas:
    # Calculo del rango intercuartilico
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Todos los datos fuera del rango intercuartilico los tratamos como missings
    data[column] = data[column].where((data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR), np.nan)



data.isnull().sum()




# Grafico para la distribucion de variables categoricas

fig, axes = plt.subplots(2, 5, figsize=(5 * 3, 4 * 3))
axes = axes.flatten()

# Grafico de barras para cada variable
for i, col in enumerate(categoricas):
    sns.countplot(data=data, x=col, palette='muted', ax=axes[i])
    axes[i].set_title(f'Distribución de "{col}"')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Recuento')
    axes[i].tick_params(axis='x', rotation=30)
    
plt.tight_layout()
plt.show()


data['Manufacturer'] = data['Manufacturer'].replace({'MERCEDES-BENZ': 'Otro', 'LEXUS': 'Otro'})
data['Drive wheels'] = data['Drive wheels'].replace({'4x4': 'Otro', 'Rear': 'Otro'})
data['Cylinders'] = data['Cylinders'].replace({1: 'Menos de 5', 2: 'Menos de 5', 3: 'Menos de 5', 4: 'Menos de 5'
                                               , 5: '5 o más', 6: '5 o más', 7: '5 o más', 8: '5 o más',
                                               9: '5 o más', 10: '5 o más', 12: '5 o más'})

missings = data.isnull().sum()
prop_missing = missings/len(data)

missing_values = pd.DataFrame({'Valores omitidos': missings,
                                  ' % del total ': prop_missing.round (2)})


# Imputacion de datos perdidos mediante KNN
imputer = KNNImputer(n_neighbors=3, metric='nan_euclidean')
df = data.copy()
data_imputed = pd.DataFrame(imputer.fit_transform(df[numericas]))
# Recuperamos los nombres de las columnas
data_imputed=data_imputed.set_axis([numericas], axis=1)


variables = list(numericas) + categoricas

data = pd.concat([data_imputed, data[categoricas]], axis=1)
data = data.set_axis(variables, axis=1)


# Aplicar MinMaxScaler para asegurarse de que los valores sean no negativos
scaler = MinMaxScaler()

# Escalamos las variables numericas
num_scaled = scaler.fit_transform(data[numericas])
num_scaled = pd.DataFrame(num_scaled)

# Unimos las columnas numericas escaladas con las categoricas
df_depurada = pd.concat([num_scaled, data[categoricas]], axis=1)

# Recuperamos los nombres de las columnas
df_depurada = df_depurada.set_axis(variables, axis=1)




# Creamos dummies para las variables categoricas
df_depurada_dummies = pd.get_dummies(df_depurada,columns=categoricas, drop_first= True)

df_depurada_dummies.head()


# Separamos la variable objetivo del dataframe
X = df_depurada_dummies.drop(columns=['Color_White'])  
y = df_depurada_dummies['Color_White']


selector = SelectKBest(score_func=chi2, k=9)
X_new = selector.fit_transform(X, y)

# Obtener los nombres de las características seleccionadas
selected_columns = X.columns[selector.get_support()]
X = df_depurada_dummies[selected_columns]

print("Variables seleccionadas:", selected_columns)

seed=123
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.30, random_state = seed, stratify=y)


# BUSQUEDA CON KERNEL LINEAL 
  
# Rango del parámetro C
param_grid_lineal = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 50, 100] } 


 #{'C': [0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75]} 


grid = GridSearchCV(SVC(kernel='linear', random_state = seed), param_grid_lineal, refit = True, cv=10, verbose = 3)
  
# Entrenamos y buscamos en TRAIN
resultados = grid.fit(X_train, y_train)

# Visualizamos los resultados para seguir buscando más profundamente

# Crear gráfico de dispersion
aux = pd.DataFrame(resultados.cv_results_)

plt.scatter(aux[['param_C']], aux[['mean_test_score']], color='b', alpha=0.9)

plt.xlabel('Parametro C')
plt.ylabel('Accuracy ')
plt.xlim(0, 2)
plt.title('Precisión media SVM en función del parametro C ')

plt.show()


print(grid.best_params_)
print(grid.best_estimator_)

# Predecimos usando el mejor modelo lineal
grid_predictions = grid.predict(X_test)
  
# Mostramos las metricas
print(classification_report(y_test, grid_predictions))


# Matriz de confusion
cm = confusion_matrix(y_test, grid_predictions)
print(cm)

# Accuracy del modelo
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)


# Metrica AUC y curva ROC
y_scores = grid.decision_function(X_test)
auc = roc_auc_score(y_test, y_scores)
print('AUC', auc)


falsospostivos, verdaderopositivo, _ = roc_curve(y_test, y_scores)

plt.figure()
plt.plot(falsospostivos, verdaderopositivo, label=f'AUC = {auc:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

best_linear_svm=grid.best_estimator_



# BUSQUEDA CON KERNEL GAUSSIANO  
param_grid_gausiano =  {'C': [1, 1.25, 1.5, 1.75, 1.9, 2, 2.1, 2.25, 2.5, 2.75, 3], 
          'gamma': [0.5 ,0.75 ,1 ,1.25, 1.5],
           'kernel': ['rbf']}  


#{'C': [0.1,0.5, 1, 2, 5, 10, 100], 
#            'gamma': [1 ,0.1 ,0.001 ,0.0001],
#           'kernel': ['rbf']} 


grid_gausiano = GridSearchCV(SVC(), param_grid_gausiano, refit = True, cv=5, verbose = 3)
  
# Entrenamos y buscamos en TRAIN
resultados = grid_gausiano.fit(X_train, y_train)


print(grid_gausiano.best_params_)
 
print(grid_gausiano.best_estimator_)

# Graficamos los resultados del modelo gaussiano
aux = pd.DataFrame(resultados.cv_results_)

# Factorizamos la variable sigma en categorias
categorias, valores_enteros = pd.factorize(aux['param_gamma'])

# Creamos el gráfico de dispersión con colores basados en las categorías de sigma
plt.scatter(aux[['param_C']], aux[['mean_test_score']], c=categorias , cmap='viridis')
plt.xlabel('Parametro C')
plt.ylabel('Accuracy ')
plt.xlim(0, 3)
plt.title('Precisión media SVM en función del parametro C ')

plt.show()


# Predecimos usando el mejor modelo gaussiano

grid_predictions = grid_gausiano.predict(X_test)
  
# Imprimimos las métricas
print(classification_report(y_test, grid_predictions))

cm = confusion_matrix(y_test, grid_predictions)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,1]+ cm[1,0]+cm[0,0])
print('accuracy' , accuracy)

y_scores = grid_gausiano.decision_function(X_test)
auc = roc_auc_score(y_test, y_scores)
print('AUC', auc)


falsospostivos, verdaderopositivo, _ = roc_curve(y_test, y_scores)

plt.figure()
plt.plot(falsospostivos, verdaderopositivo, label=f'AUC = {auc:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

best_gaussian_svm = grid_gausiano.best_estimator_

### Creacion del bagging

# El mejor modelo ha sido el lineal por terminos de accuracy y AUC
best_svm = best_linear_svm

best_svm.fit(X_train,y_train)
print('Model test Score: %.3f, ' %best_svm.score(X_test, y_test),
      'Model training Score: %.3f' %best_svm.score(X_train, y_train))


# Creamos el bagging con modelo base el ganador de antes
bagging = BaggingClassifier(best_svm, n_estimators=100, random_state=seed)

# Entrenamos el bagging
bagging.fit(X_train, y_train)

print('Model test Score: %.3f, ' %bagging.score(X_test, y_test),
      'Model training Score: %.3f' %bagging.score(X_train, y_train))



### 3. Creacion del Stacking

# Inicializar los modelos base
model_1 = LogisticRegression() 
model_2 =  KNeighborsClassifier(n_neighbors=5)  
model_3 = RandomForestClassifier() 

all_models = [('lr', model_1), ('Kn', model_2), ('rf', model_3)]

# Inicializar el modelo de stacking
stacking_model = StackingClassifier(
    estimators= all_models,
    final_estimator=  LogisticRegression(solver='liblinear') #best_svm
)


# Entrenar el modelo de stacking
stacking_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = stacking_model.predict(X_test)

# Mostrar las métricas
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print('AUC', auc)

# Obtener las características metaaprendidas por cada clasificador base
X_train_meta = stacking_model.transform(X_train)

# Evaluar el rendimiento de cada clasificador base individualmente
for (name, clf) in stacking_model.named_estimators_.items():
    y_pred_base = clf.predict(X_test)
    print(f"\nResultados del Clasificador Base {name}:")
    print(classification_report(y_test, y_pred_base))



cv = StratifiedKFold ( n_splits =5 , shuffle = True , random_state = seed )

 # Obtener predicciones de la validacion cruzada
y_pred_cv_train = cross_val_predict ( stacking_model , X_train , y_train , cv = cv ,
method ='predict')
y_pred_cv_test = cross_val_predict ( stacking_model , X_test , y_test , cv = cv ,
method ='predict')

# Calcular metricas 
accuracy = accuracy_score( y_train , y_pred_cv_train )
precision = precision_score( y_train , y_pred_cv_train )
recall = recall_score( y_train , y_pred_cv_train )
f1 = f1_score( y_train , y_pred_cv_train )

# Imprimir las métricas
print('Métricas en train:')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}\n')


accuracy = accuracy_score( y_test , y_pred_cv_test )
precision = precision_score( y_test , y_pred_cv_test )
recall = recall_score( y_test , y_pred_cv_test )
f1 = f1_score( y_test , y_pred_cv_test )


print('Metricas en test:')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')



### 4. COMPARACION DE MODELOS

svm_lineal = best_linear_svm
svm_lineal.fit(X_train, y_train)
y_pred = svm_lineal.predict(X_test)

print('Resultados SVM Lineal:\n')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred),'\n')


svm_gaussiano = best_gaussian_svm 
svm_gaussiano.fit(X_train, y_train)
y_pred = svm_gaussiano.predict(X_test)

print('Resultados SVM Gaussiano:\n')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred),'\n')


bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

print('Resultados Bagging:\n')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred),'\n')

stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
print('Resultados Stacking:\n')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))