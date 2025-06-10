import os # Proporciona funciones para interactuar con el sistema operativo.
import pandas as pd # Manipulación y análisis de datos tabulares (filas y columnas).
import numpy as np # Operaciones numéricas y matriciales.
import seaborn as sns # Visualización estadística de datos.
import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones.

# Matplotlib es una herramienta versátil para crear gráficos desde cero,
# mientras que Seaborn simplifica la creación de gráficos estadísticos.

from sklearn.decomposition import PCA # Implementación del Análisis de Componentes Principales (PCA).
from sklearn.preprocessing import StandardScaler # Estandarización de datos para análisis estadísticos.

#Definimos nuestro entorno de trabajo.
os.chdir('C:/Users/pablo/Master Big Data/Modulo 8 - Mineria de datos y modelizacion predictiva Daniel Martin/Datos') 

# Cargar un archivo Excel llamado 'datos.xlsx' en un DataFrame llamado iris.

datos = pd.read_excel('penguins.xlsx') 


# Guarda las variables  no numericas
specie = datos.iloc[:, [0]]
island = datos.iloc[:, [1]]
sex = datos.iloc[:, [6]]
print(specie)
#Elimina las variables no numericas del DataFrame 'datos'.
eliminar = ['species', 'island', 'sex']
datos = datos.drop(eliminar, axis=1)
# Genera una lista con los nombres de las variables.
variables = list(datos)
print(datos[variables].dtypes)

# Calcula las estadísticas descriptivas para cada variable y crea un DataFrame con los resultados.
estadisticos = pd.DataFrame({
    'Mínimo': datos[variables].min(),
    'Percentil 25': datos[variables].quantile(0.25),
    'Mediana': datos[variables].median(),
    'Percentil 75': datos[variables].quantile(0.75),
    'Media': datos[variables].mean(),
    'Máximo': datos[variables].max(),
    'Desviación Estándar': datos[variables].std(),
    'Varianza': datos[variables].var(),
    'Datos Perdidos': datos[variables].isna().sum()  # Cuenta los valores NaN por variable.
})
print(estadisticos)
# Calcular la matriz de covarianzas
C = datos.cov()
print(C)
# Calcula y representación de la matriz de correlación entre las 
# variables del DataFrame 'datos'.
R = datos.corr()
print(R)

# Crea una nueva figura de tamaño 10x8 pulgadas para el gráfico.
plt.figure(figsize=(10, 8))

# Genera un mapa de calor (heatmap) de la matriz de correlación 'R' utilizando Seaborn.
# 'annot=True' agrega los valores de correlación en las celdas.
# 'cmap' establece el esquema de colores (en este caso, 'coolwarm' para colores fríos y cálidos).
# 'fmt' controla el formato de los números en las celdas ('.2f' para dos decimales).
# 'linewidths' establece el ancho de las líneas que separan las celdas.
sns.heatmap(R, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)


# Estandarizamos los datos:
# Utilizamos StandardScaler() para estandarizar (normalizar) las variables.
# - StandardScaler calcula la media y la desviación estándar de las variables en 'datos' durante el ajuste.
# - Luego, utiliza estos valores para transformar 'datos' de manera que tengan media 0 y desviación estándar 1.
# - El método fit_transform() realiza ambas etapas de ajuste y transformación en una sola llamada.
# Finalmente, convertimos la salida en un DataFrame usando pd.DataFrame().
datos_estandarizadas = pd.DataFrame(
    StandardScaler().fit_transform(datos),  # Datos estandarizados
    columns=['{}_z'.format(variable) for variable in variables],  # Nombres de columnas estandarizadas
    index=datos.index  # Índices (etiquetas de filas) del DataFrame
)

# Crea una instancia de Análisis de Componentes Principales (ACP):
# - Utilizamos PCA(n_components=7) para crear un objeto PCA que realizará un análisis de componentes principales.
# - Establecemos n_components en 7 para retener el maximo de las componentes principales (maximo= numero de variables).
pca = PCA(n_components=4)

# Aplicar el Análisis de Componentes Principales (ACP) a los datos estandarizados:
# - Usamos pca.fit(datos_estandarizadas) para ajustar el modelo de ACP a los datos estandarizados.
fit = pca.fit(datos_estandarizadas)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_
print(autovalores)
# Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T, 
                            columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                            index = ['{}_z'.format(variable) for variable in variables])
print(autovectores)
# Construimos las componentes
resultados_pca = pd.DataFrame(fit.transform(datos_estandarizadas), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_estandarizadas.index)

# DEterminar el numero de componentes principales

# Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_*100

# Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
var_acumulada = np.cumsum(var_explicada)

# Crear un DataFrame de pandas con los datos anteriores y establecer índice.
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)]) 

# Imprimir la tabla
print(tabla)

# Representacion de la variabilidad explicada (Método del codo):   

def plot_varianza_explicada(var_explicada, n_components):
    """
    Representa la variabilidad explicada 
    Args:
      var_explicada (array): Un array que contiene el porcentaje de varianza explicada
        por cada componente principal. Generalmente calculado como
        var_explicada = fit.explained_variance_ratio_ * 100.
      n_components (int): El número total de componentes principales.
        Generalmente calculado como fit.n_components.
    """  
    # Crear un rango de números de componentes principales de 1 a n_components
    num_componentes_range = np.arange(1, n_components + 1)

    # Crear una figura de tamaño 8x6
    plt.figure(figsize=(8, 6))

    # Trazar la varianza explicada en función del número de componentes principales
    plt.plot(num_componentes_range, var_explicada, marker='o')

    # Etiquetas de los ejes x e y
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada')

    # Título del gráfico
    plt.title('Variabilidad Explicada por Componente Principal')

    # Establecer las marcas en el eje x para que coincidan con el número de componentes
    plt.xticks(num_componentes_range)

    # Mostrar una cuadrícula en el gráfico
    plt.grid(True)

    # Agregar barras debajo de cada punto para representar el porcentaje de variabilidad explicada
    # - 'width': Ancho de las barras de la barra. En este caso, se establece en 0.2 unidades.
    # - 'align': Alineación de las barras con respecto a los puntos en el eje x. 
    #   'center' significa que las barras estarán centradas debajo de los puntos.
    # - 'alpha': Transparencia de las barras. Un valor de 0.7 significa que las barras son 70% transparentes.
    plt.bar(num_componentes_range, var_explicada, width=0.2, align='center', alpha=0.7)

    # Mostrar el gráfico
    plt.show()
    
plot_varianza_explicada(var_explicada, fit.n_components_)

print(C)

# Crea una instancia de ACP con las dos primeras componentes que nos interesan y aplicar a los datos.
pca = PCA(n_components=2)
fit = pca.fit(datos_estandarizadas)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T, 
                            columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                            index = ['{}_z'.format(variable) for variable in variables])

print(autovectores)
# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(datos_estandarizadas), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_estandarizadas.index)

print(resultados_pca)
# Añadimos las componentes principales a la base de datos estandarizada.
datos_z_cp = pd.concat([datos_estandarizadas, resultados_pca], axis=1)

print(datos_estandarizadas)
print(datos_z_cp)
# Cálculo de las covarianzas y correlaciones entre las variables originales y las componentes seleccionadas.
# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = datos_z_cp.columns

# Guardamos el numero de componentes
n_variables = fit.n_features_in_

# Calcular la matriz de covarianzas entre veriables y componentes
Covarianzas_var_comp = datos_z_cp.cov()
Covarianzas_var_comp = Covarianzas_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]

# Calculo la matriz de correlaciones entre veriables y componentes
Correlaciones_var_comp = datos_z_cp.corr()
Correlaciones_var_comp = Correlaciones_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]
print(Correlaciones_var_comp)

# Contribucion de las componentes a la variabilidad explicada de las variables
def plot_cos2_heatmap(cosenos2):
    """
    Genera un mapa de calor (heatmap) de los cuadrados de las cargas en las Componentes Principales (cosenos al cuadrado).

    Args:
        cosenos2 (pd.DataFrame): DataFrame de los cosenos al cuadrado, donde las filas representan las variables y las columnas las Componentes Principales.

    """
    # Crea una figura de tamaño 8x8 pulgadas para el gráfico
    plt.figure(figsize=(8, 8))

    # Utiliza un mapa de calor (heatmap) para visualizar 'cos2' con un solo color
    sns.heatmap(cosenos2, cmap='Blues', linewidths=0.5, annot=False)

    # Etiqueta los ejes (puedes personalizar los nombres de las filas y columnas si es necesario)
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Cuadrados de las Cargas en las Componentes Principales')

    # Muestra el gráfico
    plt.show()

cos2 = Correlaciones_var_comp**2
print(cos2)
plot_cos2_heatmap(cos2)

# Cantidad total de variabildiad explicada de una variable 
# por el conjunto de componentes

def plot_cos2_bars(cos2):
    """
    Genera un gráfico de barras para representar la varianza explicada de cada variable utilizando los cuadrados de las cargas (cos^2).

    Args:
        cos2 (pd.DataFrame): DataFrame que contiene los cuadrados de las cargas de las variables en las componentes principales.

    Returns:
        None
    """
    # Crea una figura de tamaño 8x6 pulgadas para el gráfico
    plt.figure(figsize=(8, 6))

    # Crea un gráfico de barras para representar la varianza explicada por cada variable
    sns.barplot(x=cos2.sum(axis=1), y=cos2.index, color="blue")

    # Etiqueta los ejes
    plt.xlabel('Suma de los $cos^2$')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Varianza Explicada de cada Variable por las Componentes Principales')

    # Muestra el gráfico
    plt.show()
    

plot_cos2_bars(cos2)

# Contribuciones de cada variable en la construcción de las componentes


def plot_contribuciones_proporcionales(cos2, autovalores, n_components):
    """
    Cacula las contribuciones de cada variable a las componentes principales y
    Genera un gráfico de mapa de calor con los datos
    Args:
        cos2 (DataFrame): DataFrame de los cuadrados de las cargas (cos^2).
        autovalores (array): Array de los autovalores asociados a las componentes principales.
        n_components (int): Número de componentes principales seleccionadas.
    """
    # Calcula las contribuciones multiplicando cos2 por la raíz cuadrada de los autovalores
    contribuciones = cos2 * np.sqrt(autovalores)

    # Inicializa una lista para las sumas de contribuciones
    sumas_contribuciones = []

    # Calcula la suma de las contribuciones para cada componente principal
    for i in range(n_components):
        nombre_componente = f'Componente {i + 1}'
        suma_contribucion = np.sum(contribuciones[nombre_componente])
        sumas_contribuciones.append(suma_contribucion)

    # Calcula las contribuciones proporcionales dividiendo por las sumas de contribuciones
    contribuciones_proporcionales = contribuciones.div(sumas_contribuciones, axis=1) * 100

    # Crea una figura de tamaño 8x8 pulgadas para el gráfico
    plt.figure(figsize=(8, 8))

    # Utiliza un mapa de calor (heatmap) para visualizar las contribuciones proporcionales
    sns.heatmap(contribuciones_proporcionales, cmap='Blues', linewidths=0.5, annot=False)

    # Etiqueta los ejes (puedes personalizar los nombres de las filas y columnas si es necesario)
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Contribuciones Proporcionales de las Variables en las Componentes Principales')

    # Muestra el gráfico
    plt.show()
    
    # Devuelve los DataFrames de contribuciones y contribuciones proporcionales
    return contribuciones_proporcionales

contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2,autovalores,fit.n_components)

# Representación de las correlaciones entre variables y componentes 


def plot_corr_cos(n_components, correlaciones_datos_con_cp):
    """
    Genera un gráfico en el que se representa un vector por cada variable, usando como ejes las componentes, la orientación
    y la longitud del vector representa la correlación entre cada variable y dos de las componentes. El color representa el
    valor de la suma de los cosenos al cuadrado.
    
    Args:
        n_components (int): Número entero que representa el número de componentes principales seleccionadas.
        correlaciones_datos_con_cp (DataFrame): DataFrame que contiene la matriz de correlaciones entre variables y componentes
    """
    # Definir un mapa de color (cmap) sensible a las diferencias numéricas
    cmap = plt.get_cmap('coolwarm')  # Puedes ajustar el cmap según tus preferencias
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los cosenos al cuadrado
            sum_cos2 = correlaciones_datos_con_cp.iloc[:, i] ** 2 + correlaciones_datos_con_cp.iloc[:, j] ** 2
            
            # Crear un nuevo gráfico para cada par de componentes principales
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Dibujar un círculo de radio 1
            circle = plt.Circle((0, 0), 1, fill=False, color='b', linestyle='dotted')
            ax.add_patch(circle)
            
            # Dibujar vectores para cada variable con colores basados en la suma de los cosenos al cuadrado
            for k, var_name in enumerate(correlaciones_datos_con_cp.index):
                x = correlaciones_datos_con_cp.iloc[k, i]  # Correlación en la primera dimensión
                y = correlaciones_datos_con_cp.iloc[k, j]  # Correlación en la segunda dimensión
                
                # Seleccionar un color de acuerdo a la suma de los cosenos al cuadrado
                color = cmap(sum_cos2.iloc[k])
                
                # Dibujar el vector con el color seleccionado
                ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=color)
                
                # Agregar el nombre de la variable junto a la flecha con el mismo color
                ax.text(x, y, var_name, color=color, fontsize=12, ha='right', va='bottom')
            
            # Dibujar líneas discontinuas que representen los ejes
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            ax.set_xlabel(f'Componente Principal {i + 1}')
            ax.set_ylabel(f'Componente Principal {j + 1}')
            
            # Establecer los límites del gráfico
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            
            # Agregar un mapa de color (colorbar) y su leyenda
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])  # Evita errores de escala
            plt.colorbar(sm, ax=ax, orientation='vertical', label='cos^2')  # Agrega la leyenda
            
            # Mostrar el gráfico
            plt.grid()
            plt.show()
            
plot_corr_cos(fit.n_components, Correlaciones_var_comp)



# Nube de puntos de las observaciones en las componentes = ejes

def plot_pca_scatter(pca, datos_estandarizados, n_components):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados.

    Args:
        pca (PCA): Objeto PCA previamente ajustado.
        datos_estandarizados (pd.DataFrame): DataFrame de datos estandarizados.
        n_components (int): Número de componentes principales seleccionadas.
    """
    # Representamos las observaciones en cada par de componentes seleccionadas
    componentes_principales = pca.transform(datos_estandarizados)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los valores al cuadrado para cada variable
            # Crea un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
            plt.scatter(componentes_principales[:, i], componentes_principales[:, j])
            
            # Añade etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_estandarizados.index)
    
            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales[k, i], componentes_principales[k, j]))
            
            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')
            
            # Establece el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones en PCA')
            
            plt.show()
            
plot_pca_scatter(pca, datos_estandarizadas, fit.n_components)

# Nube de puntos de las observaciones en las componentes = ejes y correlaciones entre variables y componentes
def plot_pca_scatter_with_vectors(pca, datos_estandarizados, n_components, components_):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados
    con vectores de las correlaciones escaladas entre variables y componentes

    Args:
        pca (PCA): Objeto PCA previamente ajustado.
        datos_estandarizados (pd.DataFrame): DataFrame de datos estandarizados.
        n_components (int): Número de componentes principales seleccionadas.
        components_: Array con las componentes.
    """
    # Representamos las observaciones en cada par de componentes seleccionadas
    componentes_principales = pca.transform(datos_estandarizados)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los valores al cuadrado para cada variable
            # Crea un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
            plt.scatter(componentes_principales[:, i], componentes_principales[:, j])
            
            # Añade etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_estandarizados.index)
    
            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales[k, i], componentes_principales[k, j]))
            
            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')
            
            # Establece el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones y variables en PCA')
            
            
            # Añadimos vectores que representen las correlaciones escaladas entre variables y componentes
            fit = pca.fit(datos_estandarizados)
            coeff = np.transpose(fit.components_)
            scaled_coeff = 6 * coeff  #8 = escalado utilizado, ajustar en función del ejemplo
            for var_idx in range(scaled_coeff.shape[0]):
                plt.arrow(0, 0, scaled_coeff[var_idx, i], scaled_coeff[var_idx, j], color='red', alpha=0.5)
                plt.text(scaled_coeff[var_idx, i], scaled_coeff[var_idx, j],
                     datos_estandarizadas.columns[var_idx], color='red', ha='center', va='center')
            
            plt.show()
            
plot_pca_scatter_with_vectors(pca, datos_estandarizadas, fit.n_components, fit.components_)


# Observaciones suplementarias


# Cargar un archivo Excel llamado 'datos.xlsx' en un DataFrame llamado datos.
datos_S = pd.read_excel('penguins.xlsx') 

# Establecer la columna 'Alumno' como índice del DataFrame datos y eliminarla.

# Guarda las variables  no numericas
specie_S = datos_S.iloc[:, [0]]
island_S = datos_S.iloc[:, [1]]
sex_S = datos_S.iloc[:, [6]]
print(datos_S)
#Elimina las variables no numericas del DataFrame 'datos'.
eliminar = ['species', 'island', 'sex']
datos_S = datos_S.drop(eliminar, axis=1)

print(datos_S)


# Calcular la media y la desviación estándar de 'datos'
media_datos = datos.mean()
desviacion_estandar_datos = datos.std()

# Estandarizar 'datos_S' utilizando la media y la desviación estándar de 'datos'
datos_S_estandarizadas = pd.DataFrame(((datos_S - media_datos) / desviacion_estandar_datos))

datos_S_estandarizadas.columns = ['{}_z'.format(variable) for variable in variables]

# Agregar las observaciones estandarizadas a 'datos'
datos_sup = pd.concat([datos_estandarizadas, datos_S_estandarizadas])

# Calcular las componentes principales para el conjunto de datos combinado
componentes_principales_sup = pca.transform(datos_sup)

# Calcular las componentes principales para el conjunto de datos combinado
# y renombra las componentes
resultados_pca_sup = pd.DataFrame(fit.transform(datos_sup), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_sup.index)

# Representacion observaciones + suplementarios
plot_pca_scatter(pca, datos_sup, fit.n_components)



print(specie_S)

# Variable suplementaria
# Añadimos la variable categórica "class" en los datos
datos_componentes_sup= pd.concat([datos_sup, resultados_pca_sup], axis=1)  

extra_sup = pd.concat([specie, specie_S], axis=0)
datos_componentes_sup_extra= pd.concat([datos_componentes_sup,
                                               extra_sup], axis=1)  

#################################################################################################


def plot_pca_scatter_with_categories(datos_componentes_sup_var, componentes_principales_sup, n_components, var_categ):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados con categorías.

    Args:
        datos_componentes_sup_var (pd.DataFrame): DataFrame que contiene las categorías.
        componentes_principales_sup (np.ndarray): Matriz de componentes principales.
        n_components (int): Número de componentes principales seleccionadas.
        var_categ (str): Nombre de la variable introducida
    """
    # Obtener las categorías únicas
    categorias = datos_componentes_sup_var[var_categ].unique()

    # Iterar sobre todos los posibles pares de componentes principales
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Crear un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))
            plt.scatter(componentes_principales_sup[:, i], componentes_principales_sup[:, j],zorder=1)

            for categoria in categorias:
                # Filtrar las observaciones por categoría
                observaciones_categoria = componentes_principales_sup[datos_componentes_sup_var[var_categ] == categoria]
                # Calcular el centroide de la categoría
                centroide = np.mean(observaciones_categoria, axis=0)
                plt.scatter(centroide[i], centroide[j], label=categoria, s=230, marker='o',zorder=3)

            # Añadir etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_componentes_sup_var.index)

            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales_sup[k, i], componentes_principales_sup[k, j]))

            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')

            # Establecer el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones en PCA')

            # Mostrar la leyenda para las categorías
            plt.legend()
            plt.show()
        
plot_pca_scatter_with_categories(datos_componentes_sup_extra, componentes_principales_sup, fit.n_components, 'species')



####
# Cargar un archivo Excel llamado 'datos.xlsx' en un DataFrame llamado datos.
datos_S = pd.read_excel('penguins.xlsx') 

# Establecer la columna 'Alumno' como índice del DataFrame datos y eliminarla.

# Guarda las variables  no numericas
specie_S = datos_S.iloc[:, [0]]
island_S = datos_S.iloc[:, [1]]
sex_S = datos_S.iloc[:, [6]]
print(datos_S)
#Elimina las variables no numericas del DataFrame 'datos'.
eliminar = ['species', 'island', 'sex']
datos_S = datos_S.drop(eliminar, axis=1)

print(datos_S)


# Calcular la media y la desviación estándar de 'datos'
media_datos = datos.mean()
desviacion_estandar_datos = datos.std()

# Estandarizar 'datos_S' utilizando la media y la desviación estándar de 'datos'
datos_S_estandarizadas = pd.DataFrame(((datos_S - media_datos) / desviacion_estandar_datos))

datos_S_estandarizadas.columns = ['{}_z'.format(variable) for variable in variables]

# Agregar las observaciones estandarizadas a 'datos'
datos_sup = pd.concat([datos_estandarizadas, datos_S_estandarizadas])

# Calcular las componentes principales para el conjunto de datos combinado
componentes_principales_sup = pca.transform(datos_sup)

# Calcular las componentes principales para el conjunto de datos combinado
# y renombra las componentes
resultados_pca_sup = pd.DataFrame(fit.transform(datos_sup), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_sup.index)

# Representacion observaciones + suplementarios
plot_pca_scatter(pca, datos_sup, fit.n_components)



print(specie_S)

# Variable suplementaria
# Añadimos la variable categórica "class" en los datos
datos_componentes_sup= pd.concat([datos_sup, resultados_pca_sup], axis=1)  

extra_sup = pd.concat([island, island_S], axis=0)
datos_componentes_sup_extra= pd.concat([datos_componentes_sup,
                                               extra_sup], axis=1)  

#################################################################################################


def plot_pca_scatter_with_categories(datos_componentes_sup_var, componentes_principales_sup, n_components, var_categ):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados con categorías.

    Args:
        datos_componentes_sup_var (pd.DataFrame): DataFrame que contiene las categorías.
        componentes_principales_sup (np.ndarray): Matriz de componentes principales.
        n_components (int): Número de componentes principales seleccionadas.
        var_categ (str): Nombre de la variable introducida
    """
    # Obtener las categorías únicas
    categorias = datos_componentes_sup_var[var_categ].unique()

    # Iterar sobre todos los posibles pares de componentes principales
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Crear un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))
            plt.scatter(componentes_principales_sup[:, i], componentes_principales_sup[:, j],zorder=1)

            for categoria in categorias:
                # Filtrar las observaciones por categoría
                observaciones_categoria = componentes_principales_sup[datos_componentes_sup_var[var_categ] == categoria]
                # Calcular el centroide de la categoría
                centroide = np.mean(observaciones_categoria, axis=0)
                plt.scatter(centroide[i], centroide[j], label=categoria, s=180, marker='o',zorder=3)

            # Añadir etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_componentes_sup_var.index)

            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales_sup[k, i], componentes_principales_sup[k, j]))

            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')

            # Establecer el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones en PCA')

            # Mostrar la leyenda para las categorías
            plt.legend()
            plt.show()
        
plot_pca_scatter_with_categories(datos_componentes_sup_extra, componentes_principales_sup, fit.n_components, 'island')















# ----- Cluster -----

# Importamos las bibliotecas necesarias
import os
import pandas as pd    
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

datos.head()


# Creamos el mapa de calor (heatmap) donde se representan de manera ordenada los 
# valores observados, así como un proceso de cluster jerárquico donde se muestran 
# los diferentes pasos iterativos de unión de observaciones.

sns.clustermap(datos, cmap='coolwarm', annot=True, figsize=(50,50))
# Agregamos un título al gráfico
plt.title('Mapa de Calor')
# Etiquetamos el eje x
plt.xlabel('          longitud')
# Etiquetamos el eje y
plt.ylabel('Flor')

plt.show()

# Calculamos distancias sin estandarizar
# Calcula la matriz de distancias Euclidianas entre las observaciones
distance_matrix = distance.cdist(datos, datos, 'euclidean')
# Crea un DataFrame a partir de la matriz de distancias con los índices de df
distance_df = pd.DataFrame(distance_matrix, index=datos.index, columns=datos.index)
# La distance_matrix es una matriz 2D que contiene las distancias Euclidianas 
# entre todas las parejas de observaciones.

# Seleccionamos las primeras 5 filas y columnas de la matriz de distancias
distance_small = distance_matrix[:5, :5]
# Agregamos los nombres de índice a la matriz de distancias
distance_small = pd.DataFrame(distance_small, index=datos.index[:5], columns=datos.index[:5])
# Redondeamos los valores en la matriz de distancias
distance_small_rounded = distance_small.round(2)
print("Matriz de Distancias Redondeada:\n", distance_small_rounded)

# Representamos gráficamente la matriz de distancias

# Crea una nueva figura para el gráfico con un tamaño específico
plt.figure(figsize=(60,30))
# Genera un mapa de calor usando Seaborn
# - `distance_df`: DataFrame de pandas que contiene los datos para el mapa de calor.
# - `annot=False`: No muestra las anotaciones (valores de los datos) en las celdas del mapa.
# - `cmap="YlGnBu"`: Utiliza la paleta de colores "Yellow-Green-Blue" para el mapa de calor.
# - `fmt=".1f"`: Formato de los números en las anotaciones, en este caso no se usan.
sns.heatmap(distance_df, annot=False, cmap="YlGnBu", fmt=".1f")
# Muestra el gráfico
plt.show()




# Realizamos clustering jerárquico para obtener la matriz de enlace (linkage matrix). 
# Clustermap es una función compleja que combina un mapa de calor con dendrogramas para mostrar la agrupación de datos.
# Aquí, estamos usando el dataframe 'distance_df' que contiene las distancias euclidianas entre las observaciones.
# La opción 'cmap' establece la paleta de colores a 'YlGnBu', que es un gradiente de azules y verdes.
# La opción 'fmt' se usa para formatear las anotaciones numéricas, en este caso a un decimal.
# La opción 'annot=False' indica que no queremos anotaciones numéricas en las celdas del mapa de calor.
# La opción 'method' especifica el método de agrupamiento a usar, en este caso 'average' que calcula la media de las distancias.
linkage = sns.clustermap(distance_df, cmap="YlGnBu", fmt=".1f", annot=False, method='ward').dendrogram_row


# Estandarizamos los datos
# Inicializamos el escalador de estandarizacion
scaler = StandardScaler()

# Ajustamos y transformamos el DataFrame para estandarizar las columnas
# 'fit_transform' primero calcula la media y la desviacion estandar de cada columna para luego realizar la estandarizacion.
df_std = pd.DataFrame(scaler.fit_transform(datos), columns=datos.columns)

# Asignamos el indice del DataFrame original 'df' al nuevo DataFrame 'df_std'
# Esto es importante para mantener la correspondencia de los indices de las filas despues de la estandarizacion.
df_std.index = datos.index

# Calculamos las distancias euclidianas por pares entre las filas del DataFrame estandarizado
# 'cdist' calcula la distancia euclidiana entre cada par de filas en 'df_std'.
# Esto resulta en una matriz de distancias donde cada elemento [i, j] es la distancia entre la fila i y la fila j.
distance_std = distance.cdist(df_std, df_std, 'euclidean') 

# Imprimimos los primeros 5x5 elementos de la matriz de distancias para tener una vista previa
print(distance_std[:5,:5].round(2))


# Esto determina las dimensiones del grafico
plt.figure(figsize=(15, 8))

# Creamos un nuevo DataFrame para la matriz de distancias
# 'distance_std' se convierte en un DataFrame con indices y columnas correspondientes a 'df_std'
# Esto facilita la interpretacion del mapa de calor, ya que las filas y columnas estaran etiquetadas con los indices de 'df_std'
df_std_distance = pd.DataFrame(distance_std, index=df_std.index, columns=df_std.index)

# Generamos un mapa de calor utilizando Seaborn
# - 'df_std_distance': DataFrame que contiene los datos de distancia para visualizar.
# - 'annot=False': No muestra anotaciones numericas en cada celda del mapa de calor.
# - 'cmap="YlGnBu"': Define una paleta de colores en tonos de azul y verde para el mapa de calor.
# - 'fmt=".1f"': Formato de los numeros en las anotaciones, en este caso, un decimal.
sns.heatmap(df_std_distance, annot=False, cmap="YlGnBu", fmt=".1f")

# Mostramos el grafico resultante
plt.show()





# Realizamos clustering jerárquico para obtener la matriz de enlace (linkage matrix) sobre las distancias estandarizadas. 
linkage = sns.clustermap(df_std_distance, cmap="YlGnBu", fmt=".1f", annot=False, method='ward').dendrogram_row


# Establecemos un umbral de color para el dendrograma
color_threshold = 20

linkage_matrix = sch.linkage(df_std, method='ward')  # Puedes elegir un metodo de enlace diferente si es necesario


# Creamos el dendrograma con el umbral de color especificado
dendrogram = sch.dendrogram(linkage_matrix, labels=df_std_distance.index.tolist(), leaf_rotation=90, color_threshold=color_threshold)

# Mostramos el dendrograma
plt.show()


# Asignamos las observaciones de datos a 4 clusters

# Especificamos el numero de clusters a formar
num_clusters = 2

# Asignamos los datos a los clusters
# 'fcluster' forma clusters planos a partir de la matriz de enlace 'linkage_matrix'
# 'criterion='maxclust'' significa que formaremos un numero maximo de 'num_clusters' clusters
cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
# Mostramos las asignaciones de clusters
print("Cluster Assignments:", cluster_assignments) 

# Creamos una nueva columna 'Cluster4' y asignamos los valores de 'cluster_assignments' a ella
# Ahora 'df' contiene una nueva columna 'Cluster4' con las asignaciones de cluster
df_std['Cluster4'] = cluster_assignments

# Visualización de la distribución espacial de los clusters
# Paso 1: Realizar PCA
pca = PCA(n_components=2)  # Inicializamos PCA para 2 componentes principales
eliminar = ['Cluster4']
principal_components = pca.fit_transform(df_std.drop(eliminar, axis=1))  # Transformamos los datos a 2 componentes

fit = pca.fit(datos)
# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(df_std.drop(eliminar, axis=1)), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=df_std.index)
datos=datos.drop(eliminar, axis=1)
# Añadimos las componentes principales a la base de datos estandarizada.
df_z_cp = pd.concat([df_std, resultados_pca], axis=1)
print(df_z_cp)
# Calculo la matriz de correlaciones entre veriables y componentes
Correlaciones_var_comp = df_z_cp.corr()
Correlaciones_var_comp = Correlaciones_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]

print(Correlaciones_var_comp)


# Creamos un nuevo DataFrame para los componentes principales 2D
# Nos aseguramos de que df_pca tenga el mismo índice que datos
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=datos.index)


# Paso 2: Crear un gráfico de dispersión con colores para los clusters
plt.figure(figsize=(10, 6))  # Establecemos el tamaño del gráfico

# Recorremos las asignaciones únicas de clusters y trazamos puntos de datos con el mismo color
for cluster in np.unique(cluster_assignments):
    cluster_indices = df_pca.loc[cluster_assignments == cluster].index
    plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                df_pca.loc[cluster_indices, 'PC2'],
                label=f'Cluster {cluster}')  # Etiqueta para cada cluster
    # Anotamos cada punto con el nombre del país
    for i in cluster_indices:
        plt.annotate(i,
                     (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']), fontsize=10,
                     textcoords="offset points",  # cómo posicionar el texto
                     xytext=(0,10),  # distancia del texto a los puntos (x,y)
                     ha='center')  # alineación horizontal puede ser izquierda, derecha o centro

# Líneas de referencia para los ejes x e y
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

plt.title("Gráfico de PCA 2D con Asignaciones de Cluster")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid()
plt.show()


# Metodo del codo
# Creamos un array para almacenar los valores de WCSS para diferentes valores de K
wcss = []
    
for k in range(1, 11):  # Puedes elegir un rango diferente de valores de K
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)  # Inserta es el valor de WCSS

# Graficamos los valores de WCSS frente al numero de grupos (K) y buscamos el punto "codo"
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Metodo del Codo')
plt.xlabel('Numero de Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


# Metodo de la silueta  
# Creamos un array para almacenar los puntajes de silueta para diferentes valores de K
silhouette_scores = []
    
# Ejecutamos el clustering K-means para un rango de valores de K y calculamos el puntaje de silueta para cada K
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_std, labels)
    silhouette_scores.append(silhouette_avg)
    
# Graficamos los puntajes de silueta frente al numero de clusters (K)
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Metodo de la Silueta')
plt.xlabel('Numero de Clusters (K)')
plt.ylabel('Puntaje de Silueta')
plt.grid(True) 
plt.show()




# Analisis no jerarquico
# Configurar el número de clusters (k=4)
k = 2

# Inicializar el modelo KMeans
# 'n_clusters=k' indica el número de clusters a formar
# 'random_state=0' asegura la reproducibilidad de los resultados
# 'n_init=10' indica el número de veces que el algoritmo se ejecutará con diferentes centroides iniciales
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)


# Ajustar el modelo KMeans a los datos estandarizados
# 'df_std' es el DataFrame que contiene los datos previamente estandarizados
kmeans.fit(df_std)

# Obtener las etiquetas de los clusters para los datos
# 'kmeans.labels_' contiene la asignación de cada punto a un cluster
kmeans_cluster_labels = kmeans.labels_

# Creamos una nueva columna 'Cluster' y asignamos los valores de 'kmeans_cluster_labels' a ella
# 'Cluster4_v2' sera el nombre de la nueva columna en el DataFrame 'df'
datos['Cluster4_v2'] = kmeans_cluster_labels

# Ahora 'df' contiene una nueva columna 'Cluster4_v2' con las asignaciones de cluster
# Imprimimos los valores de la columna 'Cluster4_v2' para verificar las asignaciones de cluster
print(datos["Cluster4_v2"])







# Calculamos los valores de silueta para cada observación
silhouette_values = silhouette_samples(df_std, kmeans.labels_)
    
# Configuramos el tamaño de la figura para el gráfico
plt.figure(figsize=(8, 6))
y_lower = 10  # Inicio del margen inferior en el gráfico

# Iteramos sobre los 4 clusters para calcular los valores de silueta y dibujar el gráfico
for i in range(2):
    # Extraemos los valores de silueta para las observaciones en el cluster i
    ith_cluster_silhouette_values = silhouette_values[kmeans.labels_ == i]
    # Ordenamos los valores para que el gráfico sea más claro
    ith_cluster_silhouette_values.sort()
    
    # Calculamos donde terminarán las barras de silueta en el eje y
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    # Elegimos un color para el cluster
    color = plt.cm.get_cmap("Spectral")(float(i) / 4)
    # Rellenamos el gráfico entre un rango en el eje y con los valores de silueta
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    # Etiquetamos las barras de silueta con el número de cluster en el eje y
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Actualizamos el margen inferior para el siguiente cluster
    y_lower = y_upper + 10  # 10 para el espacio entre clusters

# Títulos y etiquetas para el gráfico
plt.title("Gráfico de Silueta para los Clusters")
plt.xlabel("Valores del Coeficiente de Silueta")
plt.ylabel("Etiqueta del Cluster")
plt.grid(True)  # Añadimos una cuadrícula para mejor legibilidad
plt.show()  # Mostramos el gráfico resultante



# Caracterizamos cada cluster
# Visualizacion de la distribucion espacial de los clusters

plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura

# Iterar a traves de las etiquetas unicas de clusters y graficar puntos de datos con el mismo color
for cluster in np.unique(kmeans_cluster_labels):
    cluster_indices = df_pca.loc[kmeans_cluster_labels == cluster].index
    plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                df_pca.loc[cluster_indices, 'PC2'],
                label=f'Cluster {cluster}')  # Poner una etiqueta para cada cluster
    
    # Anotar cada punto con su respectivo indice
    for i in cluster_indices:
        plt.annotate(i,
                     (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']),fontsize=10,
                     textcoords="offset points",  # Define como se posicionara el texto
                     xytext=(0,10),  # Define la distancia del texto a los puntos (x,y)
                     ha='center')  # Define la alineacion horizontal del texto

# Líneas de referencia para los ejes x e y
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

# Configurar el titulo y las etiquetas de los ejes del grafico
plt.title("Grafico 2D de PCA con Asignaciones de Cluster KMeans")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()  # Mostrar la leyenda
plt.grid()  # Mostrar la cuadricula
plt.show()  # Mostrar el grafico








# Caracterizamos cada cluster
# Añadimos las etiquetas como una nueva columna al DataFrame original
datos['Cluster'] = kmeans.labels_
# Ordenamos el DataFrame por la columna "Cluster"
df_sort = datos.sort_values(by="Cluster")
print(datos)
# Agrupamos los datos por la columna 'Cluster' y calculamos la media de cada grupo
# Esto proporcionará las coordenadas de los centroides de los clusters en el espacio de los datos originales
cluster_centroids_orig = df_sort.groupby('Cluster').mean()
# Redondeamos los valores para facilitar la visualización
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 1000)  # Ajusta el ancho para evitar truncamiento
cluster_centroids_orig.round(2)
# 'cluster_centroids_orig' ahora contiene los centroides de cada cluster en el espacio de los datos originales

print(df_std)


