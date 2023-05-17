import codecs
import logging
from queue import Queue
from typing import Set, List, Tuple

import numpy as np
import os

from generalized_boolean_polynoms.practice_sem_3.utils import int2binary_string


def save_minimum_polynoms_batch(writing_batch, batch_path):
    max_batch_num_monoms = max(len(t[1]) for t in writing_batch)
    actual_batch_size = len(writing_batch)
    batch_numpy = np.full((actual_batch_size, max_batch_num_monoms + 1), 4294967294,
                          dtype=np.uint32)
    for i, (f_id, m_ids) in enumerate(writing_batch):
        batch_numpy[i, 0] = f_id
        num_m_ids = len(m_ids)
        batch_numpy[i, 1:num_m_ids + 1] = list(m_ids)
    np_batch_path = os.path.join(batch_path)

    np.save(np_batch_path, batch_numpy[:actual_batch_size, :])
    writing_batch.clear()


def save_function_ids_set(found_poly_function_ids_set: Set[int], checkpoint_dir: str):
    num_ids = len(found_poly_function_ids_set)
    np_array = np.zeros(shape=num_ids, dtype=np.uint32)
    for i, f_id in enumerate(found_poly_function_ids_set):
        np_array[i] = f_id
    save_path = os.path.join(checkpoint_dir, "global_found_funcs.npy")
    np.save(save_path, np_array)


def load_function_ids_set(function_ids_numpy_path: str) -> Set[int]:
    function_ids_set = set(int(x) for x in np.load(function_ids_numpy_path))
    return function_ids_set


def save_found_functions_minimum_polys(batch: List[Tuple[int, Set[int]]], output_dir: str, batch_id: int):
    num_functions = len(batch)

    num_monoms = len(batch[0][1])

    np_batch_f_ids = np.zeros(shape=num_functions, dtype=np.uint32)
    np_batch_monom_ids = np.zeros(shape=(num_functions, num_monoms), dtype=np.uint8)

    for i, (f_id, monom_ids) in enumerate(batch):
        np_batch_f_ids[i] = f_id
        np_batch_monom_ids[i, :] = list(monom_ids)

    f_ids_path = os.path.join(output_dir, f"{batch_id}.f_ids.npy")
    monom_ids_path = os.path.join(output_dir, f"{batch_id}.monom_ids.npy")

    np.save(f_ids_path, np_batch_f_ids)
    np.save(monom_ids_path, np_batch_monom_ids)


#
# def load_found_functions_minimum_polys(f_ids_file_path: str, monom_ids_file_path: str):
#     f_ids_numpy = np.load(f_ids_file_path)
#     found_function_monom_ids_numpy = np.load(monom_ids_file_path)


def save_layer_description(checkpoint_dir: str, num_this_layer_batches: int, global_num_found_functions: int,
                           layer_num_found_functions: int):
    descr_file_path = os.path.join(checkpoint_dir, f"description.txt")
    with codecs.open(descr_file_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"{num_this_layer_batches}\t{global_num_found_functions}\t{layer_num_found_functions}\n")


def load_layer_description(checkpoint_dir: str):
    inp_path = os.path.join(checkpoint_dir, "description.txt")
    with codecs.open(inp_path, 'r', encoding="utf-8") as in_file:
        attrs = in_file.readline().strip().split('\t')
        num_this_layer_batches = int(attrs[0])
        global_num_found_functions = int(attrs[1])
        layer_num_found_functions = int(attrs[2])
    return num_this_layer_batches, global_num_found_functions, layer_num_found_functions


def load_layer_checkpoint(checkpoint_dir: str, num_vars):
    """
    Загрузка чекпоинта:
        1. Количество батчей, которое надо прочитать, глобальное число функций, число функций в этом слое,
        которые надо будет обойти
        2. На основе 1. создаются массивы
        3. Надо загрузить:
            3.1 Сет уже найденных функций
            3.2 Очередь этого слоя: либо айди, либо веткор значений, либо список мономов
    """
    val_vec_length = 2 ** num_vars
    num_this_layer_batches, global_num_found_fs, layer_num_found_fs = load_layer_description(checkpoint_dir)

    found_fs_numpy_path = os.path.join(checkpoint_dir, "global_found_funcs.npy")
    logging.info("Loading global found function ids")
    global_found_function_ids = load_function_ids_set(function_ids_numpy_path=found_fs_numpy_path)
    logging.info(f"Loaded global found function ids: {len(global_found_function_ids)}")
    assert len(global_found_function_ids) == global_num_found_fs

    polynoms_queue = Queue()
    logging.info(f"Loading layer queue. There are {num_this_layer_batches} batches")
    for batch_id in range(num_this_layer_batches):
        logging.info(f"Loading batch {batch_id}")
        f_ids_path = os.path.join(checkpoint_dir, f"{batch_id}.f_ids.npy")
        monom_ids_path = os.path.join(checkpoint_dir, f"{batch_id}.monom_ids.npy")

        batch_f_ids_numpy = np.load(f_ids_path)
        batch_monom_ids_numpy = np.load(monom_ids_path)
        assert len(batch_f_ids_numpy) == len(batch_monom_ids_numpy)
        for np_f_id, np_monom_ids in zip(batch_f_ids_numpy, batch_monom_ids_numpy):
            f_val_vec_str = int2binary_string(integer=int(np_f_id), length=val_vec_length)
            f_val_vec_numpy = np.array([int(x) for x in f_val_vec_str], dtype=int)
            monom_ids = set((int(m_id) for m_id in np_monom_ids))

            polynoms_queue.put((f_val_vec_numpy, monom_ids))
    return global_found_function_ids, polynoms_queue
