import argparse
import codecs
import os.path
from collections import Counter
from itertools import product, permutations
from typing import Dict, List, Set

import numpy as np
from queue import Queue

from tqdm import tqdm


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


def calculate_poly_eqv_classes(num_vars: int, perms: List[VariablesPermutation], output_path: str):
    value_vec_length = 2 ** num_vars
    value_vectors = (product((0, 1), repeat=value_vec_length))
    found_minimum_equivalent_poly_ids_set: Set[int] = set()
    batch_size = 1000000
    batch_strings = []
    counter = 0
    with codecs.open(output_path, 'w+', encoding="ascii") as out_file:
        for val_vec in tqdm(value_vectors, total=2 ** (value_vec_length)):
            val_vec_numpy = np.array(val_vec)

            minimum_function_id = (2 ** (2 ** num_vars)) * 10
            permutations_unique_func_ids: Set[int] = set()
            break_flag = False
            for perm in perms:
                permuted_val_vec_index = perm.permuted_val_vec_index
                permuted_val_vec_numpy = val_vec_numpy[permuted_val_vec_index]
                permuted_val_vec_str = ''.join(str(x) for x in permuted_val_vec_numpy)
                permuted_val_vec_func_id = int(permuted_val_vec_str, 2)

                if permuted_val_vec_func_id in found_minimum_equivalent_poly_ids_set:
                    break_flag = True
                    break

                if permuted_val_vec_func_id < minimum_function_id:
                    minimum_function_id = permuted_val_vec_func_id

                permutations_unique_func_ids.add(permuted_val_vec_func_id)

            if break_flag:
                continue

            if minimum_function_id not in found_minimum_equivalent_poly_ids_set:
                counter += len(permutations_unique_func_ids)
                batch_strings.append(f"{minimum_function_id}\t{len(permutations_unique_func_ids)}")
                if len(batch_strings) >= batch_size:
                    s = '\n'.join(batch_strings)
                    out_file.write(f"{s}\n")
                    batch_strings.clear()
                found_minimum_equivalent_poly_ids_set.add(minimum_function_id)
        if len(batch_strings) > 0:
            s = '\n'.join(batch_strings)
            out_file.write(f"{s}\n")

    print(f"COUNTER: {counter}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_variables', )
    args = parser.parse_args()

    num_vars = args.num_variables
    output_dir = f"res_n_{num_vars}_new"
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "eqv_classes.txt")

    # Множество двоичных наборов длины n
    input_sets = list(product((0, 1), repeat=num_vars))

    monom_positive_mask2monoms: Dict[str, List[Monom]] \
        = create_var_set2monoms_dict(num_vars=num_vars, input_sets=input_sets)

    monom_str2id, monom_id2positive_mask_str, monom_value_vectors_matrix \
        = create_monom2id(monom_positive_mask2monoms, num_vars)

    p_list = list(permutations(range(num_vars)))
    perms: List[VariablesPermutation] = []
    input_sets = np.array(input_sets)
    for i, p in enumerate(p_list):
        var_perm = VariablesPermutation(perm_np_array=np.array(p), orig_input_sets=input_sets,
                                        monom_value_vectors_matrix=monom_value_vectors_matrix,
                                        monom_str2id=monom_str2id)
        perms.append(var_perm)

    calculate_poly_eqv_classes(num_vars=num_vars, perms=perms, output_path=save_path)



if __name__ == '__main__':
    main()
