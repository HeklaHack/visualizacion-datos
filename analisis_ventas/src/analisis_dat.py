import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Cargar los datos
df = pd.read_csv('C:\\Users\\PC\\Documents\\Folder VSCode\\analisis_ventas\\data\\datos.csv')

# Mostrar las primeras filas
print("Primeros registros:")
print(df.head())

# Limpieza de datos
# Verificar y manejar los valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Eliminar filas con valores nulos en las columnas críticas (por ejemplo, 'producto', 'fecha', 'cantidad')
df = df.dropna(subset=['producto', 'fecha', 'cantidad'])

# Eliminar duplicados
df = df.drop_duplicates()

# Verificar si se eliminaron correctamente los duplicados y los nulos
print("\nDatos después de limpiar:")
print(df.isnull().sum())
print(df.duplicated().sum())

# Convertir la columna 'fecha' a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Crear nuevas columnas de 'año' y 'mes' para análisis temporal
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month

# Crear la columna ingresos (cantidad * precio)
df['ingresos'] = df['cantidad'] * df['precio']

# Análisis exploratorio básico: Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Análisis de correlaciones
corr_matrix = df[['cantidad', 'precio', 'ingresos']].corr()
print("\nMatriz de correlación:")
print(corr_matrix)

# Visualización de la matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Estándarización de precios y cantidades (si se quiere comparar la magnitud de variables diferentes)
scaler = StandardScaler()
df['cantidad_scaled'] = scaler.fit_transform(df[['cantidad']])
df['precio_scaled'] = scaler.fit_transform(df[['precio']])

# Visualización de histogramas para la distribución de los datos
plt.figure(figsize=(10,6))
sns.histplot(df['cantidad'], bins=30, kde=True, color='skyblue')
plt.title('Distribución de la Cantidad Vendida')
plt.xlabel('Cantidad')
plt.ylabel('Frecuencia')
plt.show()

# Análisis de ventas a lo largo del tiempo con descomposición de la serie temporal
df_ventas = df.groupby('fecha').agg({'cantidad': 'sum', 'precio': 'mean'})
df_ventas['ingresos'] = df_ventas['cantidad'] * df_ventas['precio']

# Descomposición de la serie temporal para extraer tendencia, estacionalidad y residuos
result = seasonal_decompose(df_ventas['ingresos'], model='additive', period=12)
result.plot()
plt.show()

# Análisis de ventas por producto
ventas_por_producto = df.groupby('producto').agg({'cantidad': 'sum', 'precio': 'mean'})
ventas_por_producto['ingresos'] = ventas_por_producto['cantidad'] * ventas_por_producto['precio']

# Visualización de ventas por producto (Top 10 productos)
top_10_productos = ventas_por_producto.nlargest(10, 'ingresos')
plt.figure(figsize=(10,6))
top_10_productos['ingresos'].plot(kind='bar', color='orange')
plt.title('Top 10 Productos con más Ingresos')
plt.xlabel('Producto')
plt.ylabel('Ingresos')
plt.xticks(rotation=45)
plt.show()

# Análisis de ventas por región (con un enfoque adicional de ventas estacionales por región)
ventas_por_region = df.groupby(['region', 'mes']).agg({'cantidad': 'sum', 'precio': 'mean'})
ventas_por_region['ingresos'] = ventas_por_region['cantidad'] * ventas_por_region['precio']

# Graficar la evolución de los ingresos por región a lo largo de los meses
plt.figure(figsize=(12,8))
sns.lineplot(data=ventas_por_region.reset_index(), x='mes', y='ingresos', hue='region', marker='o')
plt.title('Evolución de los Ingresos por Región a lo Largo del Año')
plt.xlabel('Mes')
plt.ylabel('Ingresos')
plt.legend(title='Región')
plt.show()

# Análisis de outliers: identificación de productos con ventas anómalas
q1 = df['ingresos'].quantile(0.25)
q3 = df['ingresos'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['ingresos'] < (q1 - 1.5 * iqr)) | (df['ingresos'] > (q3 + 1.5 * iqr))]

print("\nProductos con ingresos atípicos (outliers):")
print(outliers[['producto', 'ingresos']])

# Gráfico de dispersión para analizar la relación entre cantidad y precio
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='cantidad', y='precio', color='purple', alpha=0.6)
plt.title('Relación entre Cantidad y Precio')
plt.xlabel('Cantidad')
plt.ylabel('Precio')
plt.show()
