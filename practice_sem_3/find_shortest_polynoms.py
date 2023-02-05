from itertools import product
from typing import Dict, List

import numpy as np
from queue import Queue

"""
0 -> -x
1 -> x
2 - 1
"""


class Monom:
    def __init__(self, monom_mask, input_sets):
        self.monom_mask = monom_mask
        self.func_vector = self.calculate_monom_function(monom_mask=monom_mask, input_sets=input_sets)
        self.monom_str = self.__str__()


    def calculate_monom_function(self, monom_mask, input_sets):
        num_vars = len(monom_mask)
        func = np.ones(shape=(len(input_sets)), dtype=np.int)
        for i, inp_s in enumerate(input_sets):
            for var_id in range(num_vars):
                if monom_mask[var_id] == 0 and inp_s[var_id] == 1:
                    func[i] = 0
                    break
                elif monom_mask[var_id] == 1 and inp_s[var_id] == 0:
                    func[i] = 0
                    break
        print(self.__str__(), func)
        return func

    def __str__(self):
        s = ""
        has_var = False
        for i, var_val in enumerate(self.monom_mask):
            if var_val == 0:
                s += f"(-x_{i + 1})"
                has_var = True
            elif var_val == 1:
                s += f"(x_{i + 1})"
                has_var = True
        if not has_var:
            return "1"
        return s

    def __repr__(self):
        return self.monom_str


def create_var_set2monoms_dict(num_vars, input_sets) -> Dict[str, List[Monom]]:
    # 0 - отрицание, 1 - переменная, 2 - константа
    var_set2monoms = {}
    monom_mask_values = (0, 1, 2)
    e_2 = (0, 1)
    for i, monom_tuple in enumerate(product(monom_mask_values, repeat=num_vars)):
        monom_mask_str = ''.join((str(x) for x in monom_tuple))
        monom_positive_mask_str = monom_mask_str.replace("0", "1")
        print(i, monom_tuple, monom_mask_str, monom_positive_mask_str)
        if var_set2monoms.get(monom_positive_mask_str) is None:
            var_set2monoms[monom_positive_mask_str] = []
        m = Monom(monom_mask=monom_tuple, input_sets=input_sets)
        var_set2monoms[monom_positive_mask_str].append(m)
    return var_set2monoms


def create_monom2id(var_set2monoms: Dict[str, List[Monom]], num_vars):
    """
    :param var_set2monoms: Dict {Monom's variables positive mask : List of Monom objects}
    :param num_vars:
    :return:
    """
    # s = set()
    assert num_vars == len(list(var_set2monoms.keys())[0])
    num_values = int(2 ** (2 ** num_vars))
    print("num_values", num_values)

    monom_value_vectors = []
    monom_strings = []
    for var_set, monoms in var_set2monoms.items():
        for mon in monoms:
            mon_str = mon.monom_str
            mon_vector = mon.func_vector

            monom_strings.append(mon_str)
            monom_value_vectors.append(mon_vector)

    monom_value_vectors_matrix = np.stack(monom_value_vectors)
    print("CCCC", monom_value_vectors_matrix.shape)
    monom_str2id = {m_str: i for i, m_str in enumerate(monom_strings)}


    return monom_str2id, monom_value_vectors_matrix


def main():

    num_vars = 2
    # Множество двоичных наборов длины n
    input_sets = list(product((0, 1), repeat=num_vars))


    var_set2monoms = create_var_set2monoms_dict(num_vars=num_vars, input_sets=input_sets)
    var_sets_list = list(var_set2monoms)

    for k, v in var_set2monoms.items():
        print(f"AA {k} {v}")
    monom_str2id, monom_value_vectors_matrix = create_monom2id(var_set2monoms, num_vars)

    """
    1. Сгруппировать мономы по подмножествам переменных
    
    3. Обход:
        3.1. Клонировать список добавленных ранее мономов
        3.2. Обработать данный моном
        3.3. Пройтись по словарю {подмножество переменных: список мономов}
        3.4. Если монома с такой маской нет в текущем полиноме, тогда 
        3.5. Новый список мономов добавить в очередь
    """


    monoms_queue = Queue()
    for var_set in var_sets_list:
        monoms = var_set2monoms[var_set]
        for m in monoms:
            monoms_queue.put(m)

    expected_num_functions = int(2 ** (2 ** num_vars))  # TODO: -1?
    print(f"Expected num function: {expected_num_functions}")
    num_found_functions = 0

    added_monom_strs_list = set()
    zero_value_vector = np.zeros(shape=(int((2 ** num_vars))), dtype=np.int)
    while num_found_functions != expected_num_functions:
        filtered_monoms = {}
        for var_set_str, monoms_list in var_set2monoms.items():
            filtered_monoms_list = [mon for mon in monoms_list if mon.monom_str not in added_monom_strs_list]
            filtered_monoms[var_set_str] = filtered_monoms_list

        pass
        # TODO

    # TODO


if __name__ == '__main__':
    main()
