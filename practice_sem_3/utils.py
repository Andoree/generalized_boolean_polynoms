import codecs
import os
from typing import Dict, Tuple, List

import numpy as np


def read_monom_id2str(monom_id2str_path: str) -> Dict[int, str]:
    monom_id2str = {}
    with codecs.open(monom_id2str_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            monom_id, monom_str = line.strip().split('\t')
            monom_id = int(monom_id)

            monom_id2str[monom_id] = monom_str
    return monom_id2str


def read_shortest_poly_monom_ids(poly_id2monom_ids_path: str) -> Dict[int, Tuple[int]]:
    poly_id2monom_ids = {}
    with codecs.open(poly_id2monom_ids_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            poly_id, monom_ids = line.strip().split('\t')
            poly_id = int(poly_id)
            monom_ids = tuple(int(m_id) for m_id in monom_ids.split(','))
            poly_id2monom_ids[poly_id] = monom_ids
    return poly_id2monom_ids


def int2binary_string(integer: int, length: int):
    bin_str = bin(integer)[2:]
    bin_str = '0' * (max(length - len(bin_str), 0)) + bin_str

    return bin_str


def get_longest_poly_monom_ids(minimum_poly2monom_ids: Dict[int, Tuple[int]]) -> Dict[int, Tuple[int]]:
    longest_polys: Dict[int, Tuple[int]] = {}
    max_length = max(len(t) for t in minimum_poly2monom_ids.values())
    for poly_id, monom_ids in minimum_poly2monom_ids.items():
        poly_length = len(monom_ids)
        if poly_length == max_length:
            longest_polys[poly_id] = monom_ids
    return longest_polys


def load_poly_id2min_poly_id_from_directory(directory: str):
    poly_id2min_poly_id: Dict = {}
    for filename in os.listdir(directory):
        attrs = filename.split('.')
        assert len(attrs) == 4
        # num_ones = int(attrs[1])
        # batch_id = int(attrs[2])
        assert attrs[3] == "npy"
        file_path = os.path.join(directory, filename)
        np_batch = np.load(file_path)
        assert np_batch.shape[-1] == 2
        assert len(np_batch.shape) == 2
        for f_id, min_f_id in np_batch:
            poly_id2min_poly_id[int(f_id)] = int(min_f_id)
    return poly_id2min_poly_id


def load_poly_id2min_poly_id_numpy_from_directory(directory: str, num_functions):
    np_array = np.zeros(shape=num_functions, dtype=np.uint32)
    for filename in os.listdir(directory):
        attrs = filename.split('.')
        assert len(attrs) == 4
        assert attrs[3] == "npy"
        file_path = os.path.join(directory, filename)
        np_batch = np.load(file_path)
        assert np_batch.shape[-1] == 2
        assert len(np_batch.shape) == 2
        for f_id, min_f_id in np_batch:
            np_array[int(f_id)] = int(min_f_id)
    return np_array


def load_poly_id2min_poly_monom_ids_numpy(npy_file_path: str, max_monom_id: int) -> Dict[int, List]:
    np_array = np.load(npy_file_path)
    (num_functions, max_num_monoms) = np_array.shape
    function_id2monom_ids = {}
    for poly_row in np_array:
        func_id = int(poly_row[0])
        min_poly_monom_ids = [int(x) for x in poly_row[1:] if int(x) <= max_monom_id]
        function_id2monom_ids[func_id] = min_poly_monom_ids
    return function_id2monom_ids
