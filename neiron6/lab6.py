import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

np.random.seed(42)
time_steps = 1200
base_signal = 0.3 * np.sin(0.04 * np.arange(time_steps))
trend = np.linspace(0, 0.5, time_steps)
noise = 0.05 * np.random.normal(0, 1, time_steps)
spikes = np.zeros(time_steps)
spikes[300] = 1.0
spikes[700] = -0.8
vibration = base_signal + trend + noise + spikes
df = pd.DataFrame({'Vibration': vibration})

# визуализация синтетического временного ряда вибрации
plt.figure(figsize=(12, 5))
plt.plot(df['Vibration'], label='Синтетическая вибрация')
plt.title('Синтетический временной ряд вибрации подшипника')
plt.xlabel('Временные шаги')
plt.ylabel('Уровень вибрации')
plt.legend()
plt.grid(True)
plt.show()

# нормализация
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# создание обучающей выборки
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = X_train.reshape(-1, time_step, 1)
X_test = X_test.reshape(-1, time_step, 1)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# обучение
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# график потерь
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Обучение')
plt.plot(history.history['val_loss'], label='Валидация')
plt.title('График потерь модели (с Dropout)')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)
plt.show()

# предсказание
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# график предсказаний
plt.figure(figsize=(10, 5))
plt.plot(y_test_real, label='Фактическая вибрация')
plt.plot(predicted, label='Предсказанная вибрация')
plt.title('Сравнение предсказанных и реальных значений вибрации (улучшенная модель)')
plt.xlabel('Время')
plt.ylabel('Уровень вибрации')
plt.legend()
plt.grid(True)
plt.show()

# метрики
mse = mean_squared_error(y_test_real, predicted)
mae = mean_absolute_error(y_test_real, predicted)
r2 = r2_score(y_test_real, predicted)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
