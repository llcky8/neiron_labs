import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# 1. загрузка и подготовка данных
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

scaler = StandardScaler()
X = scaler.fit_transform(X)
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. необученная модель
model_untrained = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
model_untrained.compile(optimizer=Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
loss_untrained, acc_untrained = model_untrained.evaluate(X_test, y_test, verbose=0)
print(f"Результаты необученной сети — Потеря: {loss_untrained:.4f}, Точность: {acc_untrained * 100:.2f}%")

# 3. основная модель
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# 4.логи потерь по батчам
class BatchLossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

batch_logger = BatchLossLogger()

# 5. обучение
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8,
    verbose=0,
    callbacks=[batch_logger]
)

# 6. оценка модели
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Эпоха 20 — Средняя потеря: {loss:.4f}, Точность: {accuracy * 100:.2f}%")

# 7. график потерь на обучении по батчам
plt.figure(figsize=(8, 4))
plt.plot(batch_logger.batch_losses)
plt.title('График изменения потерь на обучении по батчам')
plt.xlabel('Номер батча')
plt.ylabel('Потеря')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. график изменения функции потерь на тесте по эпохам
plt.figure(figsize=(8, 4))
plt.plot(history.history['val_loss'], label='Потери на тесте')
plt.title('Потери на тестовых данных по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.grid(True)
plt.tight_layout()
plt.show()