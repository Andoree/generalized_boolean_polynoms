import argparse
import codecs
import logging
import os.path
from collections import Counter
from itertools import product, permutations
from typing import Dict, List, Set, Tuple

import numpy as np
from queue import Queue

from tqdm import tqdm

from generalized_boolean_polynoms.practice_sem_3.io import save_minimum_polynoms_batch, save_layer_description, \
    save_found_functions_minimum_polys
from generalized_boolean_polynoms.practice_sem_3.monoms import Monom, VariablesPermutation
from generalized_boolean_polynoms.practice_sem_3.utils import load_poly_id2min_poly_id_from_directory, \
    int2binary_string, load_poly_id2min_poly_id_numpy_from_directory

"""
    0 -> -x
    1 -> x
    2 - 1
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_vars', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--input_p_id2min_p_id_dir', type=str,
                        default=f"D:/AnRoot/University/VMK/сем_4_диплом/res_n_4_new")
    parser.add_argument('--output_dir', type=str, default=f"res_new/save_ckpt/res_n_4_new")
    args = parser.parse_args()
    num_vars = args.num_vars
    val_vec_length = 2 ** num_vars
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    input_p_id2min_p_id_dir = args.input_p_id2min_p_id_dir

    output_monom_id2str_path = os.path.join(output_dir, "monom_id2str.txt")
    output_checkpoint_dir = os.path.join(output_dir, "checkpoint/")
    output_minimum_monoms_numpy_dir = os.path.join(output_dir, "numpy_polys/")
    if not os.path.exists(output_minimum_monoms_numpy_dir) and output_minimum_monoms_numpy_dir != '':
        os.makedirs(output_minimum_monoms_numpy_dir)
    logging.info("Loading poly_id2min_poly_id")
    poly_id2min_poly_id = load_poly_id2min_poly_id_numpy_from_directory(input_p_id2min_p_id_dir,
                                                                        num_functions=2 ** (2 ** num_vars))
    logging.info("Poly_id2min_poly_id is loaded")
    # for p_id, mi_p_id in poly_id2min_poly_id.items():
    #     s1 = int2binary_string(integer=p_id, length=2 ** num_vars)
    #     s2 = int2binary_string(integer=mi_p_id, length=2 ** num_vars)
    #     print(s1, s2)

    # Множество двоичных наборов длины n
    input_sets = list(product((0, 1), repeat=num_vars))
    # l = 0
    # for i in tqdm(range(4 * (10 ** 9)), total=4 * (10 ** 9)):
    #     l += 1

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

    with codecs.open(output_monom_id2str_path, 'w+', encoding="utf-8") as out_file:
        for monom_str, monom_id in monom_str2id.items():
            out_file.write(f"{monom_id}\t{monom_str}\n")

    # poly_value_vector_str2monom_ids: Dict[str, List[int]] = {}
    # found_poly_str_value_vectors = set()
    found_poly_function_ids_set: Set[int] = set()
    two_degrees = np.array([2 ** (val_vec_length - i - 1) for i in range(val_vec_length)], dtype=np.int)
    print(f"two_degrees {two_degrees}")

    p_list = list(permutations(range(num_vars)))
    perms: List[VariablesPermutation] = []
    input_sets = np.array(input_sets)
    for i, p in enumerate(p_list):
        var_perm = VariablesPermutation(perm_np_array=np.array(p), orig_input_sets=input_sets,
                                        monom_value_vectors_matrix=monom_value_vectors_matrix,
                                        monom_str2id=monom_str2id)

        perms.append(var_perm)
    # perms = perms[1:]
    print(monom_value_vectors_matrix)
    num_found_functions = 0
    unique_min_poly_ids_to_find = set(int(x) for x in poly_id2min_poly_id)
    expected_num_functions = len(unique_min_poly_ids_to_find)

    print(f"Expected num functions: {expected_num_functions}")
    all_possible_positive_masks_set = set(monom_positive_mask2monoms.keys())

    monoms_queue = Queue()
    monoms_queue.put((np.zeros(shape=2 ** num_vars, dtype=np.int), set()), )

    writing_batch: List[Tuple[int, Set]] = []
    update_counter = 0
    batch_size = args.batch_size

    found_poly_function_ids_set.add(0)
    batch_counter = 0
    current_layer_batch_counter = 0
    layer_num_found_functions = 0
    # pbar = tqdm(total=expected_num_functions, miniters=10**6, maxinterval=2000)
    # pbar = tqdm(total=expected_num_functions, )
    # current_layer_poly_tuples = [(set(), np.zeros(shape=2 ** num_vars, dtype=int))]
    current_queue_depth = 1
    current_layer_queue_list = []
    current_checkpoint_dir = os.path.join(output_checkpoint_dir, f"checkpoint_{current_queue_depth}/")
    if not os.path.exists(current_checkpoint_dir):
        os.makedirs(current_checkpoint_dir)
    while len(found_poly_function_ids_set) != expected_num_functions:
        # Беру полином из очереди. Раз он в очереди, то он уже минимальный
        (current_polynom_func_vector, current_monom_ids) = monoms_queue.get()


        current_monoms_positive_masks = set(monom_id2positive_mask_str[idx] for idx in current_monom_ids)

        allowed_positive_masks = all_possible_positive_masks_set.difference(current_monoms_positive_masks)

        for new_monom_positive_mask in allowed_positive_masks:
            # Пытаюсь добавить в имеющийся полином новый моном, маска которого не встречалась
            for monom in monom_positive_mask2monoms[new_monom_positive_mask]:
                new_monom_func_vector = monom.func_vector
                # mid = monom_str2id[monom.monom_str]
                # new_monom_func_vector = monom_value_vectors_matrix[mid]

                new_poly_func_vector = (current_polynom_func_vector + new_monom_func_vector) % 2
                new_poly_function_id = int((two_degrees * new_poly_func_vector).sum())
                new_poly_min_f_id = poly_id2min_poly_id[new_poly_function_id]

                if new_poly_min_f_id in found_poly_function_ids_set:
                    continue
                monom_str = monom.monom_str

                if new_poly_function_id in unique_min_poly_ids_to_find:
                    min_poly_id2find = poly_id2min_poly_id[new_poly_function_id]

                    new_monom_id = monom_str2id[monom_str]

                    new_monom_ids_set = current_monom_ids.copy()
                    new_monom_ids_set.add(new_monom_id)

                    monoms_queue.put((new_poly_func_vector, new_monom_ids_set,))
                    found_poly_function_ids_set.add(new_poly_function_id)
                    writing_batch.append((new_poly_function_id, new_monom_ids_set))

                    len_new_function = len(new_monom_ids_set)
                    if len_new_function > current_queue_depth or len(current_layer_queue_list) > batch_size:
                        layer_num_found_functions += len(current_layer_queue_list)
                        save_found_functions_minimum_polys(batch=current_layer_queue_list,
                                                           output_dir=current_checkpoint_dir,
                                                           batch_id=current_layer_batch_counter)

                        if len_new_function > current_queue_depth:
                            current_queue_depth += 1
                            save_layer_description(checkpoint_dir=current_checkpoint_dir,
                                                   num_this_layer_batches=current_layer_batch_counter + 1,
                                                   global_num_found_functions=len(found_poly_function_ids_set),
                                                   layer_num_found_functions=layer_num_found_functions)
                            current_layer_batch_counter = 0
                            layer_num_found_functions = 0
                            current_checkpoint_dir = os.path.join(output_dir,
                                                                  f"checkpoint_{current_queue_depth}/")
                            if not os.path.exists(current_checkpoint_dir):
                                os.makedirs(current_checkpoint_dir)
                        current_layer_queue_list.clear()
                    current_layer_queue_list.append((new_poly_function_id, new_monom_ids_set))

                    if len(writing_batch) > batch_size:
                        np_batch_path = os.path.join(output_minimum_monoms_numpy_dir, f"{batch_counter}.npy")
                        batch_counter += 1
                        save_minimum_polynoms_batch(writing_batch, np_batch_path)
                else:
                    min_poly_id2find = poly_id2min_poly_id[new_poly_function_id]
                    # if min_poly_id2find in found_poly_function_ids_set:
                    #     continue
                    for perm in perms:
                        permuted_val_vec_index = perm.permuted_val_vec_index
                        permuted_new_poly_func_vector = new_poly_func_vector[permuted_val_vec_index]
                        permuted_new_poly_function_id = int((two_degrees * permuted_new_poly_func_vector).sum())

                        if permuted_new_poly_function_id != min_poly_id2find:
                            continue

                        new_monom_id = monom_str2id[monom_str]
                        new_monom_ids_set = current_monom_ids.copy()
                        new_monom_ids_set.add(new_monom_id)
                        permuted_monom_ids = set(perm.monom_id2permuted_monom_id[m_id]
                                                 for m_id in new_monom_ids_set)

                        monoms_queue.put((permuted_new_poly_func_vector, permuted_monom_ids))
                        found_poly_function_ids_set.add(permuted_new_poly_function_id)
                        writing_batch.append((permuted_new_poly_function_id, permuted_monom_ids))

                        len_new_function = len(permuted_monom_ids)
                        if len_new_function > current_queue_depth or len(current_layer_queue_list) > batch_size:
                            layer_num_found_functions += len(current_layer_queue_list)
                            print("AAA,current_layer_queue_list", current_layer_queue_list)
                            sentinel = object()
                            # for job in iter(monoms_queue.get, sentinel):
                            #     print(job)
                            print("len(monoms_queue)", monoms_queue.get())
                            # copy_queue =
                            # print("list(copy_queue)", copy_queue)

                            save_found_functions_minimum_polys(batch=current_layer_queue_list,
                                                               output_dir=current_checkpoint_dir,
                                                               batch_id=current_layer_batch_counter)

                            if len_new_function > current_queue_depth:
                                current_queue_depth += 1
                                save_layer_description(checkpoint_dir=current_checkpoint_dir,
                                                       num_this_layer_batches=current_layer_batch_counter + 1,
                                                       global_num_found_functions=len(found_poly_function_ids_set),
                                                       layer_num_found_functions=layer_num_found_functions)
                                current_layer_batch_counter = 0
                                layer_num_found_functions = 0
                                current_checkpoint_dir = os.path.join(output_dir,
                                                                      f"checkpoint_{current_queue_depth}/")
                                if not os.path.exists(current_checkpoint_dir):
                                    os.makedirs(current_checkpoint_dir)
                            current_layer_queue_list.clear()
                        current_layer_queue_list.append((permuted_new_poly_function_id, permuted_monom_ids))

                        if len(writing_batch) > batch_size:
                            np_batch_path = os.path.join(output_minimum_monoms_numpy_dir, f"{batch_counter}.npy")
                            batch_counter += 1
                            save_minimum_polynoms_batch(writing_batch, np_batch_path)

    if len(current_layer_queue_list) > 0:
        layer_num_found_functions += len(current_layer_queue_list)
        save_found_functions_minimum_polys(batch=current_layer_queue_list,
                                           output_dir=current_checkpoint_dir,
                                           batch_id=current_layer_batch_counter)
        current_queue_depth += 1
        save_layer_description(checkpoint_dir=current_checkpoint_dir,
                               num_this_layer_batches=current_layer_batch_counter + 1,
                               global_num_found_functions=len(found_poly_function_ids_set),
                               layer_num_found_functions=layer_num_found_functions)

    if len(writing_batch) > 0:
        np_batch_path = os.path.join(output_minimum_monoms_numpy_dir, f"{batch_counter}.npy")
        batch_counter += 1
        save_minimum_polynoms_batch(writing_batch, np_batch_path)


if __name__ == '__main__':
    main()
