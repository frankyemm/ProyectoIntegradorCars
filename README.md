# Análisis y Modelado de Datos de Autos

Este proyecto se centra en el análisis exploratorio de datos (EDA) y el modelado predictivo utilizando un conjunto de datos de precios de autos. Se utilizan diversas técnicas y modelos de machine learning para entender los datos y predecir precios y categorías de autos.

## Contenido

1. **Descripción del Proyecto**
2. **Instalación de Dependencias**
3. **Análisis Exploratorio de Datos**
4. **Preparación de Datos**
5. **Modelado y Evaluación**
6. **Resultados**
7. **Conclusiones**
8. **Licencia**

## 1. Descripción del Proyecto

El objetivo de este proyecto es analizar un conjunto de datos de precios de autos, identificar patrones y correlaciones, y emplear modelos para clasificar y predecir Gamma y precios de autos respectivamente. Se utilizan diferentes algoritmos, incluyendo Random Forest, Árboles de Decisión y Regresión Lineal.

## 2. Instalación de Dependencias

Asegúrate de tener instaladas las siguientes bibliotecas de Python:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
```
## 3. Análisis Exploratorio de Datos
Las etapas iniciales del proyecto incluyen:

* Lectura y visualización del conjunto de datos.
```python
df_cars = pd.read_csv('ML_car.csv')
# Verificación de tipos de datos
print(df_cars.dtypes)
```
* Verificación de valores nulos.
```python
nulls_in_df = pd.DataFrame(nulls.items(), columns=['Columna', 'Total Nulos'])
nulls_in_df
```
* Generación de visualizaciones como boxplots e histogramas.
```python
plt.figure(figsize=(10, 6))
sns.boxplot(y=df_cars['price'])
plt.title('Boxplot de Precios de Autos')
plt.xlabel('Precio')
plt.show()
```
* Análisis de correlaciones entre variables.
## 4. Preparación de Datos
* Se identifican y se corrigen valores faltantes.
* Se generan variables dummies para variables categóricas.
* Se realizan particiones de los datos en conjuntos de entrenamiento y prueba.
```python
# Generar el df con las variables dummies

df_dummies = pd.get_dummies(df_cars, drop_first=True, dtype=float)

# Generar la matriz de correlación y ordenarla de manera descendente 
# filtrar por variables que tengan correlación con el precio y sea mayor al 70%

correlation_matrix = df_dummies.corr()
correlation_with_price = correlation_matrix[['price']].sort_values(by='price', ascending=False)
filtered_correlation = correlation_with_price[correlation_with_price['price'].abs() > 0.7]
variable_above_70 = filtered_correlation[1:].index.tolist()

# Crear un mapa de calor para la correlación con 'price' y ajustar el grosor de la barra de color

plt.figure(figsize=(30, 4))
sns.heatmap(filtered_correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, vmin=0.3, vmax=1)  # Ajustar el tamaño de la barra de color
plt.title('Correlación con el Precio (mayor a 0.6)')
plt.show()
```
## 5. Modelado y Evaluación
Se implementan los siguientes modelos:

### 5.1. Clasificación
* Random Forest Classifier:
Hiperparámetros: `n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=4`
* Decision Tree Classifier:
Hiperparámetros: `max_depth=8, min_samples_split=5, min_samples_leaf=6`
### 5.2. Regresión
* Regresión Lineal:
- Se evalúa usando MSE y R².
```python
linear_model = LinearRegression()
linear_model.fit(X_train_r, y_train_r)
```
* Random Forest Regressor:
Hiperparámetros: `n_estimators=100`
```python
# Instancia y entrenamiento del modelo
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_r, y_train_r)
```
* XGBoost:
Giperparámetros: `n_estimators=100, learning_rate=0.2, max_depth=6`
```python
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.2, max_depth=6, random_state = 42)
xgb_model.fit(X_train_r, y_train_r)
```
## 6. Resultados
Los resultados incluyen:

* Matrices de confusión.
* Reportes de clasificación.
* Gráficos de dispersión para evaluar las predicciones.
- Ejemplo de Matriz de Confusión
```python
conf_matrix = confusion_matrix(y_test_c, y_pred_RFC)
print(conf_matrix)
```
## 7. Conclusiones y recomendaciones
Los modelos de clasificación y regresión proporcionan diferentes niveles de precisión en las predicciones de precios de autos y su categorización. Se debe considerar el uso de diferentes algoritmos y técnicas para mejorar el rendimiento del modelo.

Se recomienda a la empresa de autos utilizar los modelos de clasificación RandomForestClassifier y de regresión RandomForestRegressor ya que fueron los que obtuvieron mejores resultados siendo estos los siguientes:

    #Matriz de confusión RandomForestClassifier
                        [[20  3]
                        [ 0 18]]

                         precision    recall  f1-score   support

                   0       1.00      0.87      0.93        23
                   1       0.86      1.00      0.92        18

            accuracy                           0.93        41
           macro avg       0.93      0.93      0.93        41
        weighted avg       0.94      0.93      0.93        41

        ROC-AUC:  0.9975845410628019
    #Valor MSE y R^2 RandomForestRegressor
        Random Forest - MSE: 3962403.74, R^2: 0.95

## Licencia
Este proyecto está bajo la [Licencia MIT](LICENSE). Para más detalles, consulta el archivo LICENSE.

