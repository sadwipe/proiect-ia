import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the csv file
df = pd.read_csv("../data/seattle-weather.csv")

numeric_columns = ["precipitation", "temp_max", "temp_min", "wind"]
data_matrix = df[numeric_columns]

# for displaying the whole matrix:
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

# 1) Creare matrice din datele alese:
# print("Forma matricei:", data_matrix.shape)
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
