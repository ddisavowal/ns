import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=3, hidden_size=4, output_size=3, lr=0.4):
        self.W1 = np.random.uniform(-0.3, 0.3, (input_size, hidden_size))
        self.b1 = np.random.uniform(-0.3, 0.3, hidden_size)
        self.W2 = np.random.uniform(-0.3, 0.3, (hidden_size, output_size))
        self.b2 = np.random.uniform(-0.3, 0.3, output_size)
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Прямое распространение
        self.net1 = np.dot(X, self.W1) + self.b1
        self.h = self.sigmoid(self.net1)
        self.net2 = np.dot(self.h, self.W2) + self.b2
        self.y = self.sigmoid(self.net2)
        return self.y

    def backward(self, X, d):
        # Обратное распространение
        error = self.y - d
        delta2 = error * self.sigmoid_derivative(self.y)
        
        error_hidden = np.dot(delta2, self.W2.T)
        delta1 = error_hidden * self.sigmoid_derivative(self.h)

        self.W2 -= self.lr * np.dot(self.h.T, delta2)
        self.b2 -= self.lr * np.sum(delta2, axis=0)
        self.W1 -= self.lr * np.dot(X.T, delta1)
        self.b1 -= self.lr * np.sum(delta1, axis=0)
        
        # MSE
        return np.mean(error ** 2)

    def train(self, X, d, max_epochs=1000, error_limit=0.001):
        for epoch in range(max_epochs):
            y = self.forward(X)
            error = self.backward(X, d)
            print(f"Эпоха {epoch + 1}, Ошибка: {error:.6f}")
            if error < error_limit:
                print(f"Стоп на эпохе {epoch + 1}")
                break
 

if __name__ == "__main__":
    # Входные данные
    X = np.array([[0.4, -0.7, 1.3]]) 
    d = np.array([[0.3, 0.5, 0.8]]) 
    
    # обучение
    nn = NeuralNetwork(lr=0.4)
    print("Начальные веса W1:\n", nn.W1)
    print("Начальные веса W2:\n", nn.W2)
    nn.train(X, d)
    
    
    # Предсказание
    y_pred = nn.forward(X)
    print("\nПредсказанный выход:", y_pred.flatten())
    print("Эталонный выход:", d.flatten())
    print("Ошибка на выходе:", np.abs(y_pred - d).flatten())