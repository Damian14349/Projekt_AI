from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount("/content/drive")
df = pd.read_csv('/content/drive/MyDrive/AI/BrentOilPrices.csv')

# obliczanie ile jest wierszy


def Rows():
    return len(df.axes[0])

# obliczanie ile jest kolumn


def Columns():
    return len(df.axes[1])


df

print(f"Number of Rows: {Rows()}")
print(f"Number of Columns: {Columns()}")
print(f"Data types in Data Frame: \n{df.dtypes}")

x = df.Date
y = df.Price

# ustalam wielkość wykresu
plt.figure(figsize=(16, 9))
# ustalam co ma być na wykresie na osi y i x i jaki ma być kolor
plt.plot(x, y, c='green')
# ustalenie osi x
plt.xticks(np.arange(0, 9000, 400), rotation=90)
# olabelowanie osi i tytuł
plt.xlabel("Data", fontsize=20)
plt.ylabel("Cena", fontsize=20)
plt.title("Cena ropy typu brent na przestrzeni lat 1987-2022", fontsize=22)
plt.show()

max_value = df.Price.max()
min_value = df.Price.min()

# ustalam wielkość wykresu
plt.figure(figsize=(5, 5))
plt.bar(max_value, max_value, width=100, label='max oil price')
plt.bar(min_value, min_value, width=100, label='min oil price')
# usuwam niepotrzebne dane na osi x
plt.tick_params(labelbottom=False, bottom=False)
# ustalam oś y
plt.yticks(np.arange(0, 150, 10))
# labela do osi y, tytuł i ulokowanie legendy
plt.ylabel("Cena", fontsize=14)
plt.title("Minimalna i maksymalna cena ropy typu brent", fontsize=16)
plt.legend(loc='upper left')

plt.show()

# praktycznie nieczytelny z powodu zbyt dużej ilości danych wykres typu histogram
y = df.Price
# ustalenie liczby słupków na podstawie wszystkich danych z kolumny Price
bins = len(y)

plt.figure(figsize=(16, 9))
plt.hist(y, width=0.8, bins=bins)
plt.xticks(np.arange(0, 150, 2), rotation=90)
plt.yticks(np.arange(0, 30, 1))

plt.show()

# tutaj czytelniejszy histogram dzięki ograniczeniu liczby słupków
y = df.Price

plt.figure(figsize=(16, 9))
plt.hist(y, width=10, bins=10)

plt.show()

last_price = df.iloc[-1][-1]

plt.figure(figsize=(16, 9))
plt.plot(df[['Price']])
# przerywana linia wskazuje aktualną (ostatnią w zbiorze danych) wartość, w tym przypadku cenę ropy
plt.axhline(y=last_price, linestyle='dotted', color='r')
# pokazanie ile wynosi aktualna wartość
plt.text(0, last_price, "{:.0f}".format(last_price),
         color="red", va="bottom", fontsize=20)
plt.xticks(range(0, df.shape[0], 1000), df['Date'].loc[::1000], rotation=90)
plt.xlabel('Data', fontsize=20)
plt.ylabel('Cena ropy (USD)', fontsize=20)
plt.show()

# kopiuję df do nowej kopii celem skopiowania i przesunięcia wartości kolumny cena, aby można było zrobić predykcję
data = df.copy()
data['Price1'] = data['Price'].shift(-1)
data

train = data[:-9011]
test = data[-9011:]
# odrzucam ostatni wiersz przez brak danej
test = test.drop(test.tail(1).index)
test

test = test.copy()
test['baseline_pred'] = test['Price']
test

X_train = train['Price'].values.reshape(-1, 1)
y_train = train['Price1'].values.reshape(-1, 1)
X_test = test['Price'].values.reshape(-1, 1)
# Initialize the model
dt_reg = DecisionTreeRegressor(random_state=42)
# Fit the model
dt_reg.fit(X=X_train, y=y_train)
# Make predictions
dt_pred = dt_reg.predict(X_test)
# Assign predictions to a new column in test
test['dt_pred'] = dt_pred
