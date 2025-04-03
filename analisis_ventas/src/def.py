import pandas as pd

# Cargar los datos
df = pd.read_csv('C:\\Users\\PC\\Documents\\Folder VSCode\\analisis_ventas\\data\\datos.csv')

# Mostrar los primeros registros
print("Primeros registros:")
print(df.head())

# Limpieza de datos (por ejemplo, asegurarse de que no hay valores nulos)
df.isnull().sum()

# Convertir la columna 'fecha' a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Crear la columna 'ingresos' multiplicando 'cantidad' por 'precio'
df['ingresos'] = df['cantidad'] * df['precio']

# Analisis básico: ventas por producto
ventas_por_producto = df.groupby('producto').agg({'cantidad': 'sum', 'precio': 'mean'})
ventas_por_producto['ingresos'] = ventas_por_producto['cantidad'] * ventas_por_producto['precio']

print("\nVentas por producto:")
print(ventas_por_producto)

# Analisis de ventas por región
ventas_por_region = df.groupby('region').agg({'cantidad': 'sum', 'precio': 'mean'})
ventas_por_region['ingresos'] = ventas_por_region['cantidad'] * ventas_por_region['precio']

print("\nVentas por región:")
print(ventas_por_region)

# Calcular la matriz de correlación entre 'cantidad', 'precio' e 'ingresos'
corr_matrix = df[['cantidad', 'precio', 'ingresos']].corr()

# Mostrar la matriz de correlación
print("\nMatriz de correlación:")
print(corr_matrix)

# Graficar las ventas por producto
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
ventas_por_producto['ingresos'].plot(kind='bar', color='skyblue')
plt.title('Ingresos por Producto')
plt.xlabel('Producto')
plt.ylabel('Ingresos')
plt.xticks(rotation=45)
plt.show()

# Graficar las ventas por region
plt.figure(figsize=(10,6))
ventas_por_region['ingresos'].plot(kind='bar', color='lightgreen')
plt.title('Ingresos por Región')
plt.xlabel('Región')
plt.ylabel('Ingresos')
plt.xticks(rotation=45)
plt.show()

# Análisis adicional: ventas a lo largo del tiempo
ventas_por_fecha = df.groupby('fecha').agg({'cantidad': 'sum', 'precio': 'mean'})
ventas_por_fecha['ingresos'] = ventas_por_fecha['cantidad'] * ventas_por_fecha['precio']

plt.figure(figsize=(10,6))
plt.plot(ventas_por_fecha.index, ventas_por_fecha['ingresos'], marker='o', color='purple')
plt.title('Ingresos por Fecha')
plt.xlabel('Fecha')
plt.ylabel('Ingresos')
plt.xticks(rotation=45)
plt.show()
