import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

# read the csv file
df = pd.read_csv("../data/seattle-weather.csv")

numeric_columns = ["precipitation", "temp_max", "temp_min", "wind"]
data_matrix = df[numeric_columns]

# for displaying the whole matrix:
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

# 1) Creare matrice din datele alese:
# print(data_matrix)

# 2.1) Average:
# print("Average:\n", data_matrix.mean())

# 2.2) Medium value:
# print("\nMedian:\n", data_matrix.median())

# 2.3) Freq value:
# print("\nMode:\n", data_matrix.mode())

# 2.4) Range:
# range_values = data_matrix.max() - data_matrix.min()
# print("\nRange:\n", range_values)

# 2.5) Standard deviation:
# print("\nStandard Deviation:\n", data_matrix.std())



sns.set(style="whitegrid")
# 3.1) Histograms:
# data_matrix.hist(bins=20, figsize=(10, 8), color='skyblue', edgecolor='black')
# plt.suptitle("Histograme pentru variabilele numerice")
# plt.tight_layout()
# plt.show()

# 3.2) Box plots:
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=data_matrix)
# plt.title("Box Plot pentru toate variabilele")
# plt.show()

# 3.3) Scatter Plots
# plt.figure(figsize=(12, 5))

# temp_max vs temp_min
# plt.subplot(1, 2, 1)
# sns.scatterplot(x="temp_min", y="temp_max", data=data_matrix)
# plt.title("Scatter Plot: temp_min vs temp_max")

# wind vs precipitation
# plt.subplot(1, 2, 2)
# sns.scatterplot(x="wind", y="precipitation", data=data_matrix)
# plt.title("Scatter Plot: wind vs precipitation")

# plt.tight_layout()
# plt.show()


# 4) Correlations
# correlation_matrix = data_matrix.corr()

# # Heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
# plt.title("Matricea de Corelație")
# plt.tight_layout()
# plt.show()

# 5) Alegerea coloanei de iesire
target_column = "temp_max"

feature_columns = []

for col in numeric_columns:
    if col != target_column:
        feature_columns.append(col)

X = data_matrix[feature_columns]
y = data_matrix[target_column]

# 6) Data splitting
# Impartirea datelor in set de antrenare si testare:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dimensiuni seturi:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# 7) Normalization
scaler = MinMaxScaler()

# Invata pe X_train si transforma atat X_train cat si X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nExemplu X_train normalizat:")
print(X_train_scaled[:5])

# 8) Handling Class Imbalance
# Verificam dezechilibrul (969 vs 492)
df["hot_day"] = (df["temp_max"] >= 20).astype(int)
print(df["hot_day"].value_counts())

# Separă clasele
majority = df[df["hot_day"] == 0]
minority = df[df["hot_day"] == 1]

# Resample clasa minoritara
minority_upsampled = resample(minority,
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)

# Combina inapoi
df_balanced = pd.concat([majority, minority_upsampled])

print(y_train.unique())
print(y_train.unique())  # Vezi valorile unice
print(y_train.head())    # Vezi primele 5 valori

# 9) Feature selection
model = LinearRegression()
model.fit(X_train_scaled, y_train)  # Use scaled data (X_train_scaled)

# 2. Verificarea coeficientilor modelului
print("Coeficientii modelului:")
print(model.coef_)

# 3. Selectarea caracteristicilor importante (opțional pentru regresie)
selected_features = X.columns
print("\nCaracteristici:")
print(selected_features)