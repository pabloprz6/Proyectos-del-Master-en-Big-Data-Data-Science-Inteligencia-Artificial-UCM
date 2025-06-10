import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Ruta del directorio de trabajo deseado
directorio_trabajo = 'C:/Users/pablo/Master Big Data\Modulo 9 - Mineria de datos modelizacion predictivas (Juana Maria Alonso)/Tarea9'

# Cambiar al directorio de trabajo deseado
os.chdir(directorio_trabajo)
datos = pd.read_excel('TurismoCadiz.xlsx')

datos['Periodo']=datos['Periodo'].str.strip()
datos['Periodo']= pd.to_datetime(datos['Periodo'].str.replace('M', '-'), format='%Y-%m')

turismo_Cad= datos.set_index('Periodo')['Viajeros']

turismo_Cad.plot()
# Add legend and labels
plt.title('Número de viajeros en la ciudad de Cádiz')
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
# Show the plot
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
# Additive Decomposition
Additive_decomposition= seasonal_decompose(turismo_Cad, model='additive', period=12)
#Representamos los componentes de la serie obtenidos.
plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)
fig = Additive_decomposition.plot()

print(Additive_decomposition.seasonal[:12])



S_Ajustada_Est=turismo_Cad-Additive_decomposition.seasonal
# Graficar la serie original y las componentes
plt.figure(figsize=(12, 8))
# Serie original
plt.plot(turismo_Cad, label='Datos', color='gray')
# Tendencia
plt.plot(Additive_decomposition.trend, label='Tendencia', color='blue')
# Estacionalmente ajustada
plt.plot(S_Ajustada_Est, label='Estacionalmente ajustada', color='red')
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.title('Viajeros en Cádiz')
plt.legend()
plt.show()

# Extraer el año de cada fecha
#turismo_Cad['Año'] = turismo_Cad.index.year
# Crear un gráfico con colores diferentes para cada año con seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(x= turismo_Cad.index.month, y= turismo_Cad,
hue= turismo_Cad.index.year, palette='viridis')
plt.xlabel('Mes')
plt.ylabel('Estacionalidad')
plt.title('Gráfico estacional por año: Viajeros en Cádiz')
plt.legend(title='Año',loc='upper left', bbox_to_anchor=(1, 1))
plt.show()





train = turismo_Cad[:156]
test = turismo_Cad[156:]
# Add legend and labels
plt.figure(figsize=(12, 8))
# Serie original
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='yellow')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('defunciones')
# Show the plot
plt.show()




from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
# Aplicar todo tipo de suavizados
model1 = ExponentialSmoothing(train, seasonal_periods=12,trend="add",
seasonal="add", initialization_method="estimated").fit()
fcast = model1.forecast(12)
plt.figure(figsize=(12, 8))
# Series
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='yellow')
plt.plot(model1.fittedvalues, label='suavizado', color='blue')
plt.plot(fcast,color='red', label="prediciones")
plt.xlabel('Año')
plt.ylabel('Viajeros')
plt.title('Holt-Winter Aditivo')
plt.legend()

print(fcast)


from tabulate import tabulate
# Encabezados de la tabla
headers = ['Name', 'Param', 'Value', 'Optimized']

# Imprimir la tabla con formato
table_str = tabulate(model1.params_formatted, headers, tablefmt='fancy_grid')
# Mostrar la tabla formateada
print(table_str)


# Crear subgráficos
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
# Graficar en cada subgráfico
axes[0].plot(model1.level)
axes[0].set_title('Level')
axes[1].plot(model1.trend)
axes[1].set_title('Trend')
axes[2].plot(model1.season)
axes[2].set_title('Season')
# Ajustar el diseño para evitar solapamientos
plt.tight_layout()
# Mostrar los subgráficos
plt.show()





# ------ SEGUNDO TEMA ------

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Crear subgráficos para ACF y PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
# Graficar la función de autocorrelación (ACF)
plot_acf(train, lags=30, ax=ax1)
ax1.set_title('Función de Autocorrelación (ACF)')
# Graficar la función de autocorrelación parcial (PACF)
plot_pacf(train, lags=30, ax=ax2)
ax2.set_title('Función de Autocorrelación Parcial (PACF)')
# Ajustar el diseño para evitar solapamientos
plt.tight_layout()
# Mostrar los subgráficos
plt.show()


# Calcular las diferencias de orden estacional
diferencias = train.diff(12)
plt.figure(figsize=(12, 6))
plt.plot(diferencias)
plt.title('Serie en Diferencias')
plt.xlabel('Fecha')
plt.ylabel('Diferencia de viajeros')
plt.show()

# Elimina los valores faltantes que aparecen en el inicio al hacer las diferencias
diferencias = diferencias.dropna()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(diferencias, lags=30, ax=ax1)
ax1.set_title('Función de Autocorrelación (ACF) ')
plot_pacf(diferencias, lags=30, ax=ax2)
ax2.set_title('Función de Autocorrelación Parcial (PACF)')
plt.tight_layout()
plt.show()


# Ajustar el modelo ARIMA estacional (2,0,0)(0,1,1)
modelo_arima = sm.tsa.ARIMA(train, order=(2, 0, 0),
seasonal_order=(0, 1, 1, 12))

resultados = modelo_arima.fit()
# Obtener un resumen del modelo
print(resultados.summary())

# Graficar los residuos del modelo
resultados.plot_diagnostics(figsize=(12, 8))
plt.show()

print(resultados.summary())


print(resultados.mse) 

print(resultados.mae)


prediciones = resultados.get_forecast(steps=12)
predi_test=prediciones.predicted_mean
print(predi_test)

plt.figure(figsize=(12, 8))
# Serie original
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='yellow')
plt.plot(prediciones.predicted_mean, label='Predicciones', color='blue') # Corregir aquí
plt.xlabel('fecha')
plt.ylabel('Viajeros')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()


intervalos_confianza = prediciones.conf_int()
plt.figure(figsize=(12, 8))
plt.plot(intervalos_confianza['lower Viajeros'], label='UCL', color='gray')
plt.plot(intervalos_confianza['upper Viajeros'], label='LCL', color='gray')
plt.plot(predi_test, label='Predicciones', color='blue')
plt.plot(test, label='Test', color='yellow')
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()


### Suavizado vs Arima

plt.plot(test, label='Test', color='yellow')
plt.plot(fcast,color='red', label="Prediciones suavizado")
plt.plot(predi_test, label='Predicciones ARIMA', color='blue')
plt.legend()
plt.title("Comparación de Predicciones")
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
# Errores del método de suavizado
mae_suavizado = mean_absolute_error(test, fcast)
rmse_suavizado = np.sqrt(mean_squared_error(test, fcast))

# Errores del modelo ARIMA
mae_arima = mean_absolute_error(test, predi_test)
rmse_arima = np.sqrt(mean_squared_error(test, predi_test))


# Mostrar resultados
print(f"MAE Suavizado: {mae_suavizado}, RMSE Suavizado: {rmse_suavizado}")
print(f"MAE ARIMA: {mae_arima}, RMSE ARIMA: {rmse_arima}")









## ajuste automatico

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train, lags=30, ax=ax1)
ax1.set_title('Función de Autocorrelación (ACF)')
plot_pacf(train, lags=30, ax=ax2)
ax2.set_title('Función de Autocorrelación Parcial (PACF)')
plt.tight_layout()
plt.show()

diferencias = train.diff(12)
# Graficar las diferencias con matplotlib
plt.figure(figsize=(12, 6))
plt.plot(diferencias)
plt.title('Serie en Diferencias')
plt.xlabel('Fecha')
plt.ylabel('Diferencia de producción')
plt.show()

# Eliminar los valores faltantes
diferencias = diferencias.dropna()
# Crear subgráficos para ACF y PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(diferencias, lags=30, ax=ax1)
ax1.set_title('Función de Autocorrelación (ACF) ')
plot_pacf(diferencias, lags=30, ax=ax2)
ax2.set_title('Función de Autocorrelación Parcial (PACF)')
plt.tight_layout()
plt.show()


import pmdarima as pm
modelo_auto= pm.auto_arima(train, start_p=1, start_q=1,
max_p=3, max_q=3,
m=12, start_P=0, seasonal=True, d=0, D=1, trace=True,
error_action='ignore',
suppress_warnings=True, stepwise=True) # mostramos los modelos probados

print(modelo_auto.summary())

best_arima = sm.tsa.ARIMA(train, order=(1, 0, 2),
seasonal_order=(0, 1, 1, 12))
resultados_a = best_arima.fit()
# Graficar los residuos del modelo
resultados_a.plot_diagnostics(figsize=(12, 8))
plt.show()



### PREDICCIONES

prediciones_a = resultados_a.get_forecast(steps=12)
predi_test_a=prediciones_a.predicted_mean
intervalos_confianza_a = prediciones_a.conf_int()
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='yellow')
plt.plot(prediciones_a.predicted_mean, label='Predicciones', color='blue')
plt.xlabel('periodo')
plt.ylabel('GWh')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()