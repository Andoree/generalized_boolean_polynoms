import codecs
import os.path
from collections import Counter
from itertools import product, permutations
from typing import Dict, List, Set

import numpy as np
from queue import Queue

from tqdm import tqdm

"""
    0 -> -x
    1 -> x
    2 - 1
"""

"""
    1. Сгруппировать мономы по подмножествам переменных
    
    3. Обход:
        3.1. Клонировать список добавленных ранее мономов
        3.2. Обработать данный моном
        3.3. Пройтись по словарю {подмножество переменных: список мономов}
        3.4. Если монома с такой маской нет в текущем полиноме, тогда 
        3.5. Новый список мономов добавить в очередь
    В очереди должны быть:
        1. На текущий момент добавленные мономы. Наверное, лучше хранить именно их id
        2. Маски (нет отрицаний) мономов уже добавленных мономов
        3. Текущий вектор значений
    """


def create_variable_negation_monom2monom(var_id: int, monom_str2id: Dict[str, int]) -> Dict[int, int]:
    monom_id2_negated_monom_id = {}
    for monom_str, monom_id in monom_str2id.items():

        negated_var_str = f"(-x_{var_id + 1})"
        positive_var_str = f"(x_{var_id + 1})"
        if negated_var_str in monom_str:
            new_monom_str = monom_str.replace(negated_var_str, positive_var_str)
            new_monom_id = monom_str2id[new_monom_str]
        elif positive_var_str in monom_str:
            new_monom_str = monom_str.replace(positive_var_str, negated_var_str)
            new_monom_id = monom_str2id[new_monom_str]
        else:
            # print(monom_str, "||", negated_var_str,"||", positive_var_str)
            new_monom_id = monom_id
        monom_id2_negated_monom_id[monom_id] = new_monom_id
    return monom_id2_negated_monom_id


def create_monom2monom_all_negations(monom_str2id: Dict[str, int], num_variables: int) -> Dict[int, Dict[int, int]]:
    negation_dictionaries = {}
    for var_id in range(num_variables):
        monom_id2_negated_monom_id = create_variable_negation_monom2monom(var_id=var_id, monom_str2id=monom_str2id)
        negation_dictionaries[var_id] = monom_id2_negated_monom_id
    return negation_dictionaries


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
        # print(self.__str__(), func)
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


class VariablesPermutation:
    def __init__(self, perm_np_array: np.array, orig_input_sets, monom_value_vectors_matrix: np.array,
                 monom_str2id: Dict[str, int]):
        self.perm_np_array = perm_np_array
        # TODO: Мне надо понять, какие значения переменных переходят в какие?
        self.new_input_sets = orig_input_sets[:, perm_np_array]
        orig_input_sets_str = [''.join((str(x) for x in t)) for t in orig_input_sets]
        orig_input_set_str2set_id = {set_str: i for i, set_str in enumerate(orig_input_sets_str)}
        new_input_sets_strs_list = [''.join((str(x) for x in t)) for t in self.new_input_sets]
        self.permuted_val_vec_index = np.array(
            [orig_input_set_str2set_id[set_str] for set_str in new_input_sets_strs_list])

        monom_value_vector_str2monom_id = {''.join(str(x) for x in val_vec):
                                               m_id for m_id, val_vec in enumerate(monom_value_vectors_matrix)}

        self.monom_id2permuted_monom_id = {}
        for monom_str, monom_id in monom_str2id.items():
            monom_value_vec = monom_value_vectors_matrix[monom_id]
            permuted_monom_value_vec = monom_value_vec[self.permuted_val_vec_index]
            permuted_monom_value_vec_str = ''.join(str(x) for x in permuted_monom_value_vec)
            new_monom_id = monom_value_vector_str2monom_id[permuted_monom_value_vec_str]
            self.monom_id2permuted_monom_id[monom_id] = new_monom_id


def create_var_set2monoms_dict(num_vars, input_sets) -> Dict[str, List[Monom]]:
    # 0 - отрицание, 1 - переменная, 2 - константа
    var_set2monoms = {}
    monom_mask_values = (0, 1, 2)
    e_2 = (0, 1)
    for i, monom_tuple in enumerate(product(monom_mask_values, repeat=num_vars)):
        monom_mask_str = ''.join((str(x) for x in monom_tuple))
        monom_positive_mask_str = monom_mask_str.replace("0", "1")
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

    monom_value_vectors = []
    monom_strings = []
    var_set_strs_list = []
    for var_set, monoms in var_set2monoms.items():
        for mon in monoms:
            mon_str = mon.monom_str
            mon_vector = mon.func_vector

            monom_strings.append(mon_str)
            monom_value_vectors.append(mon_vector)
            var_set_strs_list.append(var_set)

    monom_value_vectors_matrix = np.stack(monom_value_vectors)

    monom_str2id = {m_str: i for i, m_str in enumerate(monom_strings)}
    monom_id2positive_mask_str = {i: var_set for i, var_set in enumerate(var_set_strs_list)}

    return monom_str2id, monom_id2positive_mask_str, monom_value_vectors_matrix


def main():
    num_vars = 4
    output_dir = f"res_n_{num_vars}_new"
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    output_monom_id2str_path = os.path.join(output_dir, "monom_id2str.txt")
    output_value_vector_str2monom_ids_path = os.path.join(output_dir, "output_value_vector2monom_ids_path.txt")
    output_value_vector_str2monom_strs_path = os.path.join(output_dir, "output_value_vector_str2monom_strs.txt")
    output_length_stats_path = os.path.join(output_dir, "output_length_stats.txt")

    # Множество двоичных наборов длины n
    input_sets = list(product((0, 1), repeat=num_vars))

    # Группируем объекты класса моном по положительной маске (из 1 и 2).
    monom_positive_mask2monoms: Dict[str, List[Monom]] \
        = create_var_set2monoms_dict(num_vars=num_vars, input_sets=input_sets)

    monom2var_set = {}
    # Строим отображение из строкового представления монома в строку-маску (из 1 и 2)
    for vs, m_list in monom_positive_mask2monoms.items():
        for mnm in m_list:
            monom2var_set[mnm.monom_str] = vs
    # monom2var_set = {: vs for vs, m_list in var_set2monoms.items()}
    var_positive_sets_list = list(monom_positive_mask2monoms.keys())

    monom_str2id, monom_id2positive_mask_str, monom_value_vectors_matrix \
        = create_monom2id(monom_positive_mask2monoms, num_vars)
    monom_id2str = {v: k for k, v in monom_str2id.items()}

    print("monom_str2id", monom_str2id)
    print("monom_positive_mask2monoms", monom_positive_mask2monoms)
    print("monom_id2positive_mask_str", monom_id2positive_mask_str)

    negation_dictionaries: Dict[int, Dict[int, int]] \
        = create_monom2monom_all_negations(monom_str2id=monom_str2id, num_variables=num_vars)
    # for monom_id, monom_str in monom_id2str.items():
    #     valvec = monom_value_vectors_matrix[monom_id]
    #     print(f"dd: {monom_str}: {valvec}")

    # for var_id, d in negation_dictionaries.items():
    #     for old_mon_id, new_mon_id in d.items():
    #         print(var_id,)
    #         print(monom_id2str[old_mon_id],)
    #         print(monom_value_vectors_matrix[old_mon_id])
    #         print(monom_id2str[new_mon_id], )
    #         print(monom_value_vectors_matrix[new_mon_id])
    #         print('--')

    # monom_id2str = {v: k for k, v in monom_str2id.items()}
    # for i, d in negation_dictionaries.items():
    #     for old_m_id, new_m_id in d.items():
    #         print(f"Var {i + 1}: {monom_id2str[old_m_id]} -> {monom_id2str[new_m_id]}")

    with codecs.open(output_monom_id2str_path, 'w+', encoding="utf-8") as out_file:
        for monom_str, monom_id in monom_str2id.items():
            out_file.write(f"{monom_id}\t{monom_str}\n")

    # poly_value_vector_str2monom_ids: Dict[str, List[int]] = {}
    # found_poly_str_value_vectors = set()
    found_poly_function_ids_set: Set[int] = set()
    monoms_queue = Queue()

    p_list = list(permutations(range(num_vars)))
    perms: List[VariablesPermutation] = []
    input_sets = np.array(input_sets)
    for i, p in enumerate(p_list):
        var_perm = VariablesPermutation(perm_np_array=np.array(p), orig_input_sets=input_sets,
                                        monom_value_vectors_matrix=monom_value_vectors_matrix,
                                        monom_str2id=monom_str2id)
        perms.append(var_perm)
    # perms = perms[1:]
    # for perm in perms:
    #     perm_np_array = perm.perm_np_array
    #     new_input_sets = perm.new_input_sets
    #     permuted_val_vec_index = perm.permuted_val_vec_index
    #     print("perm_np_array", perm_np_array)
    #     print("new_input_sets", new_input_sets)
    #     print("permuted_val_vec_index", permuted_val_vec_index)
    #     print('--')
    #
    # raise Exception

    num_found_functions = 0
    # Начальная обработка полиномов длины 1
    # with codecs.open(output_value_vector_str2monom_ids_path, 'w+', encoding="utf-8") as value_vector_str2monom_ids_file:
    # TODO: убрал
    # for var_set in var_positive_sets_list:
    #     monoms = monom_positive_mask2monoms[var_set]
    #     for m in monoms:
    #         monom_string = m.monom_str
    #         new_monom_mask = m.monom_mask
    #         monom_id = monom_str2id[monom_string]
    #         added_monom_ids = {monom_id, }
    #         new_monom_func_vector = monom_value_vectors_matrix[monom_id]
    #
    #         # new_monom_func_vector_str = ''.join((str(x) for x in new_monom_func_vector))
    #         monoms_queue.put((added_monom_ids, new_monom_func_vector))

    expected_num_functions = int(2 ** (2 ** num_vars)) - 1
    print(f"Expected num functions: {expected_num_functions}")
    all_possible_positive_masks_set = set(monom_positive_mask2monoms.keys())
    # monoms_queue_tmp = Queue()
    # next_layer_found_functions_str_value_vectors_set = set()
    monoms_queue = Queue()
    monoms_queue.put((set(), np.zeros(shape=2 ** num_vars, dtype=np.int)), )
    # writing_batch = ""
    pbar = tqdm(total=expected_num_functions, mininterval=10)
    length_stats = Counter()
    writing_batch: List[str] = []
    update_counter = 0
    with codecs.open(output_value_vector_str2monom_ids_path, 'w+', encoding="ascii") as value_vector_str2monom_ids_file:
        # next_layer_found_functions_str_value_vectors_set.clear()
        while len(found_poly_function_ids_set) != expected_num_functions:
            if pbar.n != len(found_poly_function_ids_set):
                pbar.n = len(found_poly_function_ids_set)
                update_counter += 1
                if update_counter > 100:
                    pbar.refresh()
                    update_counter = 0
            # while not monoms_queue.empty():
            (current_monom_ids, current_polynom_func_vector) = monoms_queue.get()

            current_monoms_positive_masks = set(monom_id2positive_mask_str[idx] for idx in current_monom_ids)

            allowed_positive_masks = all_possible_positive_masks_set.difference(current_monoms_positive_masks)

            for new_monom_positive_mask in allowed_positive_masks:
                monoms_list = monom_positive_mask2monoms[new_monom_positive_mask]
                for monom in monoms_list:
                    monom_str = monom.monom_str
                    new_monom_func_vector = monom.func_vector

                    new_poly_func_vector = (current_polynom_func_vector + new_monom_func_vector) % 2
                    new_poly_func_vector_str = ''.join((str(x) for x in new_poly_func_vector))

                    new_poly_function_id = int(new_poly_func_vector_str, 2)

                    if new_poly_function_id not in found_poly_function_ids_set:
                        new_monom_id = monom_str2id[monom_str]

                        assert new_monom_id not in current_monom_ids
                        new_monom_ids_set = current_monom_ids.copy()
                        new_monom_ids_set.add(new_monom_id)

                        monoms_queue.put((new_monom_ids_set, new_poly_func_vector))

                        for perm in perms:

                            permuted_val_vec_index = perm.permuted_val_vec_index
                            permuted_new_poly_func_vector = new_poly_func_vector[permuted_val_vec_index]

                            permuted_monom_ids = tuple(perm.monom_id2permuted_monom_id[m_id]
                                                       for m_id in new_monom_ids_set)

                            permuted_new_poly_func_vector_str = ''.join((str(x) for x in permuted_new_poly_func_vector))
                            # permuted_new_poly_func_vector_str_2 = ''.join((str(x) for x in permuted_func_vector_2))
                            # assert permuted_new_poly_func_vector_str == permuted_new_poly_func_vector_str_2

                            permuted_new_poly_function_id = int(permuted_new_poly_func_vector_str, 2)

                            if permuted_new_poly_function_id not in found_poly_function_ids_set:
                                writing_batch.append(f"{permuted_new_poly_function_id}"
                                                     f"\t{','.join(str(x) for x in permuted_monom_ids)}")
                                # writing_batch += f"{permuted_new_poly_function_id}" \
                                #                  f"\t{','.join(str(x) for x in permuted_monom_ids)}\n"
                                found_poly_function_ids_set.add(permuted_new_poly_function_id)

                            # for negation_var_id, monom_id2_negated_monom_id in negation_dictionaries.items():
                            #     negated_monom_ids = tuple(
                            #         monom_id2_negated_monom_id[m_id] for m_id in permuted_monom_ids)
                            #
                            #     negated_func_vector = monom_value_vectors_matrix[negated_monom_ids, :].sum(axis=0) % 2
                            #
                            #     assert len(negated_func_vector) == 2 ** num_vars
                            #     negated_func_vector_str = ''.join((str(x) for x in negated_func_vector))
                            #     negated_func_id = int(negated_func_vector_str, 2)
                            #
                            #
                            #     if negated_func_id not in found_poly_function_ids_set:
                            #         writing_batch += f"{negated_func_id}" \
                            #                          f"\t{','.join(str(x) for x in negated_monom_ids)}\n"
                            #         found_poly_function_ids_set.add(negated_func_id)

                            if len(found_poly_function_ids_set) > 250000:
                                s = '\n'.join(writing_batch)
                                value_vector_str2monom_ids_file.write(f"{s}\n")
                                writing_batch = []

                    else:
                        continue
        if len(writing_batch) > 0:
            s = '\n'.join(writing_batch)
            value_vector_str2monom_ids_file.write(f"{s}\n")
            writing_batch = []

            # filtered_monoms_list = [mon for mon in monoms_list if mon.monom_str not in added_monom_strs_list]
            # filtered_monoms[var_set_str] = filtered_monoms_list

    # length_counter = Counter()
    # for i, (value_vector_str, m_ids) in enumerate(poly_value_vector_str2monom_ids.items()):
    #     monoms = [monom_id2str[m_id] for m_id in m_ids]
    #     # print(i + 1, value_vector_str, m_ids, ' + '.join(monoms))
    #     length_counter[len(monoms)] += 1

    # print("monom_id2str", monom_id2str)

    # with codecs.open(output_length_stats_path, 'w+', encoding="utf-8") as out_file:
    #     for k, v in length_counter.items():
    #         out_file.write(f"{k}\t{v}\n")


if __name__ == '__main__':
    main()
