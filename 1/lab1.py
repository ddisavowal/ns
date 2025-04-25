import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Perceptron:
    def __init__(self, n_features, lr=0.1, max_epochs=1000):
        self.w = np.random.randn(n_features) * 0.01 
        self.b = np.random.randn() * 0.01  
        self.lr = lr 
        self.max_epochs = max_epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return (self.sigmoid(z) >= 0.5).astype(int)

    def train(self, X, y, error_limit=0.001):
        n = X.shape[0]
        for epoch in range(self.max_epochs):
            error = 0
            dw = np.zeros_like(self.w)
            db = 0
            for i in range(n):
                z = np.dot(X[i], self.w) + self.b
                pred = self.sigmoid(z)
                err = y[i] - pred
                error += err ** 2
                grad = err * pred * (1 - pred)
                dw += self.lr * grad * X[i]
                db += self.lr * grad
            self.w += dw
            self.b += db
            error /= n
            print(f"Эпоха {epoch + 1}, Ошибка: {error:.4f}")
            if error < error_limit:
                print(f"Стоп на эпохе {epoch + 1}")
                break

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

def load_spam_data():
    # Генерация датасета
    X, y = make_classification(n_samples=200, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, 
                              class_sep=2.0, random_state=42)
    # Нормализация в [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    return train_test_split(X, y, test_size=0.2, random_state=42)

def print_data(X, y, title, n_samples=15):
    print(f"\n{title}:")
    print("Подозр. слова (x1) | Частота ссылок (x2) | Класс (0=не спам, 1=спам)")
    for i in range(min(n_samples, len(y))):
        print(f"{X[i, 0]:.4f}             | {X[i, 1]:.4f}              | {y[i]}")

def print_predictions(X, y, y_pred, title, n_samples=15):
    print(f"\n{title}:")
    print("Подозр. слова (x1) | Частота ссылок (x2) | Истинный | Предсказанный")
    for i in range(min(n_samples, len(y))):
        print(f"{X[i, 0]:.4f}             | {X[i, 1]:.4f}              | {y[i]}        | {y_pred[i]}")



if __name__ == "__main__":
    # Генерация данных для обучения
    X_train, X_test, y_train, y_test = load_spam_data()
    print_data(X_train, y_train, "Обучающие данные (первые 15 примеров)")
    
    # Обучение 
    model = Perceptron(n_features=2, lr=0.1, max_epochs=100)
    model.train(X_train, y_train, error_limit=0.001)

    print(f"\nТочность на обучающей выборке: {model.accuracy(X_train, y_train):.4f}")
    print(f"Точность на тестовой выборке: {model.accuracy(X_test, y_test):.4f}")

    y_pred = model.predict(X_test)
    print_predictions(X_test, y_test, y_pred, "Предсказания на тестовых данных (первые 15 примеров)")
    