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

def create_data(file, n):
    for _ in range(n):
        file.write(str(np.random.normal(0, 1)) + "\n")

def arma(noise, p, q):
    global A, B
    n = len(noise)
    m = max(p, q)

    y, residuals = np.zeros(n), np.zeros(n)
    y[:m] = noise[:m]

    for t in range(m, n):
        ar_part = A[0] + sum([A[i] * y[t-i] for i in range(1, p+1)])  # AR частина
        ma_part = sum([B[i-1] * noise[t-i] for i in range(1, q+1)])  # MA частина
        y[t] = noise[t] + ar_part + ma_part
        residuals[t] = noise[t] # залишкт (похибки)

    return y, residuals

def regressor_matrix(y, res, p, q):
    n = len(y)
    m = max(p, q)
    x = np.zeros((n - m, p + q + 2))
    
    e_offset = p + 1
    
    for t in range(m, n):
        x[t-m, 0] = 1
        for i in range(1, p+1):
            x[t-m, i] = y[t-i]
        
        x[t-m, e_offset] = res[t]
        for i in range(1, q+1):
            x[t-m, e_offset+i] = res[t-i]
    
    return x

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
    global A, B
    A = [0.05, 0.45, 0.09, -0.38]
    B = [0.6, 0.4, 0]

    # Зчитуємо послідовність білого шуму з файлу
    noise = []
    with open('./data/time_series.txt', 'r') as file:
        noise = [float(value.strip()) for value in file.readlines()]
    
    for p in range(1, 4):
        for q in range(1, 4):
            print(f"ARMA({p},{q})")
            y, residuals = arma(noise, p, q)
            y_dependent = y[max(p, q):]
            X = regressor_matrix(y, residuals, p, q)
            
            theta = mnk(y_dependent, X)
            a0 = theta[0]
            A_ = theta[1:p+1]
            R2_ = theta[p+1]
            B_ = theta[p+2:]
            
            # Обчислюємо коефіцієнт детермінації R^2
            # R2 = calculateR2(y, y_dependent)
            # print(f"R^2: {R2}")

            # Обчислюємо критерій Акайке
            N = len(y)
            p, q = 3, 2
            AIC = calculateAIC(residuals, p, q, N)
            print(f"Akaike criterion (AIC): {AIC}\n")

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == "--generate-data":
        with open("./data/time_series.txt", "w") as file:
            create_data(file, int(sys.argv[2]))
    else:
        main()