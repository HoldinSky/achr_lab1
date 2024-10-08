import numpy as np
import math
import random
import sys

# Функція для оцінки MНК
def leastSquares(y, x):
    n = len(y)
    sumX = sum(x)
    sumY = sum(y)
    sumXY = sum([x[i] * y[i] for i in range(n)])
    sumXX = sum([x[i] ** 2 for i in range(n)])

    b = (n * sumXY - sumX * sumY) / (n * sumXX - sumX ** 2)
    a = (sumY - b * sumX) / n
    return a, b  # Повертаємо оцінені коефіцієнти

# Функція для обчислення коефіцієнта детермінації R^2
def calculateR2(y, y_hat):
    meanY = np.mean(y)
    ss_total = sum((yi - meanY) ** 2 for yi in y)
    ss_residual = sum((y[i] - y_hat[i]) ** 2 for i in range(len(y)))

    return 1 - (ss_residual / ss_total)

# Функція для обчислення критерію Акаіке (IKA)
def calculateAIC(e, p, q, N):
    sumError = sum(ei ** 2 for ei in e)
    n = p + q + 1
    return N * math.log(sumError / N) + 2 * n

# Функція для генерації одного нормально розподіленого числа за допомогою рівномірно розподілених чисел
def generateNormalRandom():
    sum_vals = sum(random.uniform(-1, 1) for _ in range(12))
    return sum_vals - 6  # Повертаємо нормально розподілене число

def create_data(file, n):
    for _ in range(n):
        file.write(str(generateNormalRandom()) + "\n")

def arma(y, p, q):
    global P, Q
    n = len(y)
    m = max(p, q)

    y_hat, e = np.zeros(n), np.zeros(n)
    y_hat[:m] = y[:m]

    for t in range(m, n):
        ar_part = np.sum([P[i] * y[t-i-1] for i in range(p)])  # AR частина
        ma_part = np.sum([Q[i] * e[t-i-1] for i in range(q)])  # MA частина
        y_hat[t] = ar_part + ma_part
        e[t] = y[t] - y_hat[t]

    return y_hat, e

def mnk(y, x):
    return np.linalg.inv(x.T @ x) @ x.T @ y

def rmnk(y, X, beta=10):
    N, n_params = X.shape

    P = beta * np.eye(n_params)
    theta = np.zeros(n_params)
    for i in range(len(y)):
        x_i = X[i, :].reshape(-1, 1)
        y_i = y[i]
        
        P = P - (P @ x_i @ x_i.T @ P) / (1 + x_i.T @ P @ x_i)
        theta = theta + (P @ x_i).flatten() * (y_i - x_i.T @ theta)
    return theta

# Головна програма
def main():
    # Задаємо коефіцієнти
    global P, Q
    P = [0.05, 0.45, 0.09, -0.38]
    Q = [0.6, 0.4, 0]

    # Зчитуємо часовий ряд з файлу
    y = []
    with open('./data/time_series.txt', 'r') as file:
        y = [float(value.strip()) for value in file.readlines()]

    for p in range(1,4):
        for q in range(1,4):
            print(f"ARMA({p},{q})")
            y_eval, e = arma(y, p, q)    

            # Обчислюємо коефіцієнт детермінації R^2
            R2 = calculateR2(y, y_eval)
            print(f"R^2: {R2}")

            # Обчислюємо критерій Акайке
            N = len(y)
            p, q = 3, 2
            AIC = calculateAIC(e, p, q, N)
            print(f"Akaike criterion (AIC): {AIC}\n")

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == "--generate-data":
        with open("./data/time_series.txt", "w") as file:
            create_data(file, int(sys.argv[2]))
    else:
        main()