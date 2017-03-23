# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    # error = np.empty(len(x))
    # for i in range(0, len(x)):
    #     error[i] = (y[i] - polynomial(x[i], w)) ** 2
    error = [y[i] - polynomial(x[i], w) for i in range(0, len(x))]
    return np.mean(np.square(error))


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    matrix = np.empty([len(x_train),M+1])
    for x in range(0,len(x_train)):
        for i in range(0,M+1):
            matrix[x][i] = x_train[x]**i
    return matrix


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    matrix = design_matrix(x_train, M)
    matrixT = np.transpose(matrix)
    w = np.linalg.inv(matrixT @ matrix) @ matrixT @ y_train
    error = mean_squared_error(x_train, y_train, w)
    return (w, error)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    matrix = design_matrix(x_train, M)
    matrixT = np.transpose(matrix)
    I = np.eye(matrix.shape[1])
    w = np.linalg.inv(matrixT @ matrix + regularization_lambda * I) @ matrixT @ y_train
    error = mean_squared_error(x_train, y_train, w)
    return (w, error)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    W = [least_squares(x_train, y_train, m) for m in M_values ]
    E = [(w, t_err, mean_squared_error(x_val, y_val, w)) for (w, t_err) in W ]
    return min(E, key= lambda e:e[2])


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    Wl = [(regularized_least_squares(x_train, y_train, 7, l), l) for l in lambda_values]
    El = [(w, t_err, mean_squared_error(x_val, y_val, w), l) for ((w, t_err), l) in Wl]
    return min(El, key= lambda el:el[2])