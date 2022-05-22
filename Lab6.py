"""
Формируется матрица F следующим образом: скопировать в нее А и если в В количество простых чисел в нечетных столбцах, чем произведение чисел по периметру С, то поменять местами С и В симметрично,
иначе С и В поменять местами несимметрично. При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то вычисляется выражение: AA^T – KF, иначе вычисляется выражение (A^-1 +G^-(F-1))*K,где G-нижняя треугольная матрица, полученная из А.
 Выводятся по мере формирования А, F и все матричные операции последовательно..
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
def is_simple_number(x):
    divider = 2
    if x == 0 or x < 0:
        return False
    while divider < (x**0.5):
        if x % divider == 0:
            return False
        divider += 1
    return True

def print_matrix(Matrix, matrix_name, timetime):
    print(f"Матрица {matrix_name} промежуточное время = {round(timetime, 2)} seconds.")
    for i in Matrix:  # Делаем перебор всех строк матрицы
        for j in i:  # Деребираем все элементы в строке
            print("%5d" % j, end=" ")
        print()
print("\n-------Результат работы программы-------")
try:
    matrix_size = int(input("Введите количество строк (столбцов) квадратной матрицы больше 4 : "))
    while matrix_size < 4 or matrix_size > 100:
        matrix_size = int(input("Вы ввели неверное число\nВведите количество строк (столбцов) квадратной матрицы больше 4 :"))
    K = int(input("Введите число К="))
    start = time.time()
    A = np.zeros((matrix_size, matrix_size))
    F = np.zeros((matrix_size, matrix_size))
    time_next = time.time()
    for i in range(matrix_size):  # Формируем матрицу А
        for j in range(matrix_size):
            A[i][j] = np.random.randint(-10, 10)
    time_prev = time_next
    time_next = time.time()
    print_matrix(A, "A", time_next - time_prev)

    for i in range(matrix_size):    # Формируем матрицу F, копируя из матрицы А
        for j in range(matrix_size):
            F[i][j] = A[i][j]

    submatrix_size = matrix_size // 2    # Фазмерность подматрицы
    C = np.zeros((submatrix_size, submatrix_size))
    B = np.zeros((submatrix_size, submatrix_size)) # Формируем матрицу Е

    for i in range(submatrix_size):
        for j in range(submatrix_size):
            B[i][j] = F[i][submatrix_size + j]

    for i in range(0, submatrix_size):  # формируем подматрицу E
        for j in range(0, submatrix_size):
            C[i][j] = F[submatrix_size + i][matrix_size - submatrix_size + j]
    kol_vo = 0
    multiplication = 1
    for i in range(0, submatrix_size, 1):
        for j in range(0, submatrix_size, 1):  # Обработка подматрицы Е
            if is_simple_number(B[i][j]) == True and j % 2 == 0:
                kol_vo += 1 # Подсчёт кол-ва нулей в нечётных столбцах

    for j in range(submatrix_size):
        multiplication *= C[0][j]
        # print(f"{multiplication}:'{C[0][j]}'")
    for i in range(1, submatrix_size-1):
        multiplication *= C[i][-1]
        # print(f"{multiplication}:'{C[i][-1]}'")
    for j in range(submatrix_size-1, 0-1, -1):
        multiplication *= C[-1][j]
        # print(f"{multiplication}:'{C[-1][j]}'")
    for i in range(submatrix_size-2, 0, -1):
        multiplication *= C[i][0]
        # print(f"{multiplication}:'{C[i][0]}'")
    print(multiplication)

    if kol_vo > multiplication:
        print("Случай 1")
        for i in range(0, submatrix_size + matrix_size % 2):  # Меняем подматрицы B и C местами симметрично
            for j in range(submatrix_size + matrix_size % 2, matrix_size):
                F[i][j], F[matrix_size - i - 1][j] = F[matrix_size - i - 1][j], F[i][j]
    else:
        print("Случай 2")
        for j in range(0, matrix_size // 2 + matrix_size % 2 - 1, 1):
            for i in range(matrix_size // 2):
                F[i][matrix_size // 2 + matrix_size % 2 + j], F[matrix_size // 2 + matrix_size % 2 + i][matrix_size // 2 + matrix_size % 2 + j] = F[matrix_size // 2 + matrix_size % 2 + i][matrix_size // 2 + matrix_size % 2 + j], F[i][matrix_size // 2 + matrix_size % 2 + j]
    time_prev = time_next
    time_next = time.time()
    print_matrix(B, "B", time_next - time_prev)

    time_prev = time_next
    time_next = time.time()
    print_matrix(F, "F", time_next - time_prev)

    if np.linalg.det(A) == 0 or np.linalg.det(F) == 0: # A или F вырожденая матрица,т.е вычислить нельзя
        print("A или F вырожденая матрица,т.е вычислить нельзя")
    elif np.linalg.det(A) > sum(F.diagonal()):
        A = ((np.dot(A, np.transpose(A))) - np.dot(K, F))
    else:
        A = np.dot((np.linalg.matrix_power(A, -1) + np.tril(A) - np.linalg.matrix_power(F, -1)), K)

    time_prev = time_next
    time_next = time.time()
    print_matrix(A, "A", time_next - time_prev)

    time_prev = time_next
    time_next = time.time()
    print_matrix(F, "F", time_next - time_prev)
    print(f"\nProgramm time {time.time() - start}")

    # 1 пример (matplotlib.pyplot)
    plt.title('Exampl', fontsize=15)
    plt.xlabel("Values", fontsize=13)
    plt.ylabel("Numb", fontsize=13)
    plt.grid()
    plt.gcf().canvas.manager.set_window_title("Вывод")
    for j in range(matrix_size):
        plt.plot([i for i in range(matrix_size)], A[j][::], marker='4')
    plt.show()

    # 2 пример (seaborn, pandas)
    fig, ax = plt.subplots(figsize=(4, 9))
    ax.matshow(A)
    plt.show()

    # 3 пример (matplotlib.pyplot)
    sb.catplot(data=pd.DataFrame(A), kind="box")
    plt.show()


except FileNotFoundError:
    print("\nФайл text.txt в директории проекта не обнаружен.")
