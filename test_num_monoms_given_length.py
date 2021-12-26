import random
import time
from itertools import product
from typing import Dict, Tuple, List

from generalized_boolean_polynoms.classes import Polynom
from generalized_boolean_polynoms.transform import generalized_polynom_to_zhegalkin

FACTORIALS_MEMORY = [1]


def factorial(n):
    if len(FACTORIALS_MEMORY) <= n:
        for i in range(len(FACTORIALS_MEMORY), n + 1):
            FACTORIALS_MEMORY.append(FACTORIALS_MEMORY[i - 1] * i)
    return FACTORIALS_MEMORY[n]


def binomial_coefficient(n, k):
    n_factorial = factorial(n)
    k_factorial = factorial(k)
    n_minus_k_factorial = factorial(n - k)
    return n_factorial // (k_factorial * n_minus_k_factorial)


def generate_sampling_ranges(n) -> Dict[int, Tuple[int, int]]:
    num_monoms = int(3 ** n)
    low = 0
    ranges_dict = {}
    for i in range(num_monoms + 1):
        low += 1
        num_polynoms_of_length_i = binomial_coefficient(n=num_monoms, k=i)
        high = low + num_polynoms_of_length_i - 1
        values_range = (low, high)
        ranges_dict[i] = values_range
        low = high
        # assert (high - low + 1) == num_polynoms_of_length_i
    return ranges_dict


def generate_possible_monoms_of_n_variables(n: int) -> List[Tuple[int]]:
    possible_monom_mask_values = (-1, 0, 1)
    possible_monoms = [tuple(t) for t in (product(possible_monom_mask_values, repeat=n))]
    # print('aa', possible_monoms)
    return possible_monoms


def sample_polynom(n, sampling_ranges_dict: Dict[int, Tuple[int]], possible_monoms: Tuple[Tuple[int]], seed=42):

    num_polynoms_of_n_variables = int(2 ** (3 ** n))
    # Генерируем натуральное число из диапазона [1; 2^(3^n)]
    random_int = random.randint(1, num_polynoms_of_n_variables)
    num_monoms = -1
    # Ищем, в диапазон какого числа мономов попало сгенерированное натуральное число
    for i, (low, high) in sampling_ranges_dict.items():
        if low <= random_int <= high:
            num_monoms = i
    # print(random_int, sampling_ranges_dict)
    assert num_monoms != -1
    # Выбираем фиксированное число случайных монов без повторений
    sampled_monoms = random.sample(possible_monoms, num_monoms, )
    # print('ss', sampled_monoms)
    polynom = Polynom(monoms=sampled_monoms)
    return polynom


def generate_zero_polynom(n, sampling_ranges_dict, possible_monoms, seed=42):
    random.seed(seed)
    poly = sample_polynom(n, sampling_ranges_dict, possible_monoms, seed=seed)
    zhegalkin_poly = generalized_polynom_to_zhegalkin(gen_poly=poly)

    # i = 0
    start_time = time.time()
    while len(zhegalkin_poly) != 0:
        poly = sample_polynom(n, sampling_ranges_dict, possible_monoms, seed=seed)
        zhegalkin_poly = generalized_polynom_to_zhegalkin(gen_poly=poly)
        # i += 1
        # if i % 10000 == 0:
        #     print(i, zhegalkin_poly)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print(time_elapsed)
    return poly


def main():

    sampling_ranges_dict = generate_sampling_ranges(n=4)
    possible_monoms = generate_possible_monoms_of_n_variables(n=4)
    for i in range(100):
        poly_ = generate_zero_polynom(4, sampling_ranges_dict, possible_monoms, seed=42)
        print(i, poly_)
# for i in range(4):
#     print('--')
#
#     monoms = generate_possible_monoms_of_n_variables(i)
#     print(i, len(monoms))
#     for mon in monoms:
#         print(mon)

# pos_monoms = ((0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 2, 3), (3, 5, 3), (2, 4, 0, 1))
# for i in range(len(pos_monoms)):
#     sample = random.sample(pos_monoms, i)
#     print(i, sample)
# n = 4
# num_monoms = int(3 ** 4)
# print("Всего мономов:", num_monoms)
# d = {}
# for i in range(num_monoms + 1):
#     d[i] = binomial_coefficient(n=num_monoms, k=i)
#     print(i, d[i])
# print(sum(d.values()))
# print(2 ** (3 ** n))
"""
Итак, я хочу посчитать статистику по полиномам заданной длины (заданного количества мономов)
1. Сколько всего возможных мономов от n переменных?
    3 ^ n
    n = 1: 1, x , -x
    n = 2: -x-y, -xy, -x, -y, 1, y, x-y, x, xy 



"""

#
# for i in range(20):
#     print(i, factorial(i))
if __name__ == '__main__':
    main()