from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# класс "Формальный нейрон"
class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = random.random()
        self.learning_rate = 0.1
        print(f"[Создан нейрон] Входов: {input_size}, Веса: {self.weights}, Смещение: {round(self.bias, 3)}")


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if self.sigmoid(summation) >= 0.5 else 0

    def train(self, training_inputs, labels, epochs=30):
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                summation = np.dot(inputs, self.weights) + self.bias
                output = self.sigmoid(summation)
                error = label - output
                total_error += abs(error)
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
            print(f"Эпоха {epoch + 1}/{epochs}, суммарная ошибка: {round(total_error, 2)}")

# функция подготовки изображения
def prepare_image(path):
    img = Image.open(path).resize((200, 200))
    img = np.array(img)[:, :, :3]
    avg_r = np.mean(img[:, :, 0]) / 255
    avg_g = np.mean(img[:, :, 1]) / 255
    return img, np.array([avg_r, avg_g])

# генерация тренировочных данных
train_data = []
train_labels = []

# красные изображения (метка 0)
for _ in range(5):
    red_img = np.zeros((200, 200, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255
    avg_r = np.mean(red_img[:, :, 0]) / 255
    avg_g = np.mean(red_img[:, :, 1]) / 255
    train_data.append(np.array([avg_r, avg_g]))
    train_labels.append(0)

# зелёные изображения (метка 1)
for _ in range(5):
    green_img = np.zeros((200, 200, 3), dtype=np.uint8)
    green_img[:, :, 1] = 255
    avg_r = np.mean(green_img[:, :, 0]) / 255
    avg_g = np.mean(green_img[:, :, 1]) / 255
    train_data.append(np.array([avg_r, avg_g]))
    train_labels.append(1)

# обучение нейрона
perceptron = Neuron(input_size=2)
perceptron.train(np.array(train_data), np.array(train_labels), epochs=30)

# проверка на внешних изображениях
image_paths = [
    ("red1.jpg", "красный"),
    ("green1.jpg", "зелёный"),
    ("red2.jpg", "красный"),
    ("green2.jpeg", "зелёный"),
    ("red3.png", "красный"),
    ("green3.jpeg", "зелёный")
]

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for idx, (img_path, expected_label) in enumerate(image_paths):
    img, features = prepare_image(img_path)
    prediction = perceptron.predict(features)
    predicted_label = "красный" if prediction == 0 else "зелёный"

    ax = axes[idx // 3, idx % 3]
    ax.imshow(img)
    ax.set_title(f"Ожид.: {expected_label}\nПредсказ.: {predicted_label}")
    ax.axis('off')

plt.tight_layout()
plt.show()
