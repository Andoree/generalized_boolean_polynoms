import argparse
import gc
import math
import os.path
from itertools import product, permutations, combinations
from typing import Dict, List, Set

import numpy as np

from generalized_boolean_polynoms.practice_sem_3.trashcan.find_shortest_polynoms_for_diploma_v2 import Monom, \
    VariablesPermutation


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


def create_monom2id(var_set2monoms: Dict[str, List[Monom]], num_vars, ):
    """
    :param var_set2monoms: Dict {Monom's variables positive mask : List of Monom objects}
    :param num_vars:
    :return:
    """

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


def calculate_poly_eqv_classes(num_vars: int, perms: List[VariablesPermutation], num_ones_min, num_ones_max,
                               output_path: str):
    value_vec_length = 2 ** num_vars
    target_val_vec_numpy = np.zeros(shape=value_vec_length, dtype=np.int)
    permuted_val_vec_numpy = np.zeros(shape=value_vec_length, dtype=np.int)
    i_range = list(range(value_vec_length))
    permutations_unique_func_ids: Set[int] = set()
    processed_function_ids: Set[int] = set()
    batch_func_id2min_func_id_dict: Dict[int, int] = {}
    batch_size = 10 ** 6
    dirname = os.path.dirname(output_path)
    filename = os.path.basename(output_path).split('.')[0]
    ext = os.path.basename(output_path).split('.')[1]
    counter = 0

    numpy_batch = np.zeros(shape=(batch_size + num_vars ** num_vars, 2), dtype=np.uint32)
    max_batch_size = len(numpy_batch)
    print(max_batch_size)

    for num_ones in range(num_ones_min, num_ones_max + 1):
        print(f"\nProcessing num ones: {num_ones}\n")
        output_path_ = f"{dirname}/{filename}.{num_ones}.{ext}"
        gc.collect()
        batch_counter = 0
        # with codecs.open(output_path_, 'w+', encoding="ascii") as out_file:
        progress_bar_length = \
            math.factorial(value_vec_length) // \
            (math.factorial(value_vec_length - num_ones) * math.factorial(num_ones))
        for one_indices in combinations(i_range, num_ones):
            # mininterval=10.0,
            one_indices = list(one_indices)
            target_val_vec_numpy[:] = 0
            target_val_vec_numpy[one_indices] = 1
            minimum_function_id = (2 ** (2 ** num_vars)) * 10
            permutations_unique_func_ids.clear()

            break_flag = False
            for perm in perms:
                permuted_val_vec_index = perm.permuted_val_vec_index
                permuted_val_vec_numpy[:] = target_val_vec_numpy[permuted_val_vec_index]

                permuted_val_vec_str = ''.join(str(x) for x in permuted_val_vec_numpy)
                permuted_val_vec_func_id = int(permuted_val_vec_str, 2)

                if permuted_val_vec_func_id in processed_function_ids:
                    break_flag = True
                    break

                if permuted_val_vec_func_id < minimum_function_id:
                    minimum_function_id = permuted_val_vec_func_id

                permutations_unique_func_ids.add(permuted_val_vec_func_id)
            target_val_vec_numpy[one_indices] = 0
            if not break_flag:
                for func_id in permutations_unique_func_ids:
                    assert batch_func_id2min_func_id_dict.get(func_id) is None
                    batch_func_id2min_func_id_dict[func_id] = minimum_function_id
                processed_function_ids.update(permutations_unique_func_ids)
            if len(batch_func_id2min_func_id_dict) > batch_size:
                actual_batch_size = len(batch_func_id2min_func_id_dict)
                for i, (id_1, id_2) in enumerate(batch_func_id2min_func_id_dict.items()):
                    numpy_batch[i][0] = id_1
                    numpy_batch[i][1] = id_2

                assert i + 1 == actual_batch_size
                numpy_batch_path = f"{dirname}/{filename}.{num_ones}.{batch_counter}.npy"
                np.save(numpy_batch_path, numpy_batch[:actual_batch_size, :])

                counter += actual_batch_size

                # TODO: Сохранить numpy
                batch_func_id2min_func_id_dict.clear()
                batch_counter += 1
                if batch_counter % 5 == 0:
                    gc.collect()
        # TODO: В конце батч сбросить на диск
        if len(batch_func_id2min_func_id_dict) > 0:
            actual_batch_size = len(batch_func_id2min_func_id_dict)
            for i, (id_1, id_2) in enumerate(batch_func_id2min_func_id_dict.items()):
                numpy_batch[i][0] = id_1
                numpy_batch[i][1] = id_2
            assert i + 1 == actual_batch_size
            numpy_batch_path = f"{dirname}/{filename}.{num_ones}.{batch_counter}.npy"
            np.save(numpy_batch_path, numpy_batch[:actual_batch_size, :])
            batch_counter += 1
        # if len(batch_func_id2min_func_id_dict) > 0:
        #     # s = '\n'.join(f'{t[0]},{t[1]}' for t in batch_func_id2min_func_id_dict.items())
        #     # out_file.write(f"{s}\n")
        #     counter += len(batch_func_id2min_func_id_dict)
        #     batch_func_id2min_func_id_dict.clear()
        #     batch_counter += 1
    print(f"Processed : {counter}")


def main(args):
    num_vars = args.num_vars
    # output_dir = f"D:/AnRoot/University/VMK/сем_4_диплом/res_n_{num_vars}_new"
    output_dir = os.path.join(args.output_dir, f"res_n_{num_vars}_new/")
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "eqv_classes.txt")
    value_vec_length = 2 ** num_vars
    num_ones_min = args.num_ones_min
    num_ones_max = args.num_ones_max
    assert num_ones_min <= value_vec_length
    assert num_ones_max <= value_vec_length

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

    calculate_poly_eqv_classes(num_vars=num_vars, perms=perms, output_path=save_path,
                               num_ones_min=num_ones_min, num_ones_max=num_ones_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_vars', type=int, )
    parser.add_argument('--num_ones_min', type=int, )
    parser.add_argument('--num_ones_max', type=int, )
    parser.add_argument('--output_dir', type=str, )
    args = parser.parse_args()
    main(args)
