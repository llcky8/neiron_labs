import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

data_path = "C:/Users/llcky/Documents/datasets/kmnist/"

x_train = np.load(data_path + "kmnist-train-imgs.npz")['arr_0']
y_train = np.load(data_path + "kmnist-train-labels.npz")['arr_0']
x_test = np.load(data_path + "kmnist-test-imgs.npz")['arr_0']
y_test = np.load(data_path + "kmnist-test-labels.npz")['arr_0']

# предобработка
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

# One-hot кодирование меток
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
class_names = [f"Класс {i}" for i in range(10)]

# создание модели CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# компиляция и обучение модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2)

# оценка точности на тесте
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Точность на тестовой выборке: {test_acc:.4f}")

# график точности
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('KMNIST: Точность на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.savefig("kmnist_accuracy.png")

# визуализация предсказаний
np.random.seed(42)
indices = np.random.choice(len(x_test), 10, replace=False)
sample_images = x_test[indices]
true_labels = np.argmax(y_test_cat[indices], axis=1)
pred_labels = np.argmax(model.predict(sample_images), axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(sample_images[i].squeeze(), cmap='gray')
    ax.set_title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}", fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig("kmnist_predictions.png")

# синтетика
def generate_synthetic_data(num_samples=1000, img_size=32):
    images, labels = [], []
    for _ in range(num_samples):
        img = np.zeros((img_size, img_size), dtype=np.float32)
        center = (img_size // 2, img_size // 2)
        radius = np.random.randint(10, 14)
        cv2.circle(img, center, radius, 1.0, thickness=-1)

        label = np.random.choice([0, 1])  # 0 - годное, 1 - брак
        if label == 1:
            if np.random.rand() < 0.5:
                p1 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
                p2 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
                cv2.line(img, p1, p2, 0.0, 1)
            if np.random.rand() < 0.5:
                noise = np.random.rand(img_size, img_size) < 0.05
                img[noise] = 0.0
            if np.random.rand() < 0.5:
                pts = [(np.random.randint(0, img_size), np.random.randint(0, img_size)) for _ in range(3)]
                for i in range(2):
                    cv2.line(img, pts[i], pts[i + 1], 0.0, 1)

        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

train_imgs, train_lbls = generate_synthetic_data(1000)
test_imgs, test_lbls = generate_synthetic_data(200)

train_imgs = np.expand_dims(train_imgs, -1)
test_imgs = np.expand_dims(test_imgs, -1)
train_lbls_cat = to_categorical(train_lbls, 2)
test_lbls_cat = to_categorical(test_lbls, 2)

model_synth = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model_synth.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_synth = model_synth.fit(train_imgs, train_lbls_cat, epochs=10, batch_size=32, validation_split=0.2)
test_loss_synth, test_acc_synth = model_synth.evaluate(test_imgs, test_lbls_cat)
print(f"Точность на тестовой выборке (синтетика): {test_acc_synth:.2f}")

# график точности (синтетика)
plt.figure()
plt.plot(history_synth.history['accuracy'], label='Train')
plt.plot(history_synth.history['val_accuracy'], label='Validation')
plt.title('Синтетика: Точность на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.savefig("synthetic_accuracy.png")

# визуализация предсказаний
new_imgs, new_lbls = generate_synthetic_data(10)
new_imgs_exp = np.expand_dims(new_imgs, -1)
new_preds = np.argmax(model_synth.predict(new_imgs_exp), axis=1)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(new_imgs[i], cmap='gray')
    ax.set_title(f"True: {new_lbls[i]}, Pred: {new_preds[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("synthetic_predictions.png")