import argparse
import codecs
import itertools
import os
from typing import Dict, Tuple

import numpy as np


def read_monom_index(inp_path: str) -> Dict[int, Tuple[int]]:
    monom_id2monom_mask = {}
    with codecs.open(inp_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            monom_id = int(attrs[0])
            monom_mask = tuple(int(x) for x in attrs[1].split(','))
            monom_id2monom_mask[monom_id] = monom_mask
    return monom_id2monom_mask


def read_value_vector2minimum_poly(inp_path) -> Dict[str, Tuple[int]]:
    val_vector_str2monom_ids = {}
    with codecs.open(inp_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            val_vector_str = attrs[0]
            monom_ids = tuple(int(x) for x in attrs[1].split(','))
            val_vector_str2monom_ids[val_vector_str] = monom_ids
    return val_vector_str2monom_ids


def filter_longest_polynoms(val_vector_str2poly_monom_ids: Dict[str, Tuple[int]]) -> Dict[str, Tuple[int]]:
    maximum_length = max((len(t) for t in val_vector_str2poly_monom_ids.values()))
    longest_val_vector_str2poly_monom_ids = \
        {key: t for key, t in val_vector_str2poly_monom_ids.items() if len(t) == maximum_length}

    return longest_val_vector_str2poly_monom_ids



def main():
    """
    Что мне нужно сделать?
        ВАРИАНТ 1
        1. Разложить полином от 4 переменных?
            1.1 Для этого надо для каждого длинного полинома сделать следующее:
            1.1.1 Разложить его n раз по числу переменных на полиномы от 3-х переменных
            1.1.2 Для получившихся двух полиномов от трёх переменных, надо проверить, что они являются длиннейшими в своём классе?
            1.1.3 Отдельно запомнить те случаи, когда получается как длиннейший и когда не получаются
        2. Попробовать построить самые длинные полиномы от 4 переменных как полиномы от 3 переменных. Для этого надо:
            2.1 Пройтись по всем парам самых длинных полиномов от 3-х переменных
            2.2. Попробовать для каждого такого полинома добавить ещё одну переменную. Только...
            2.2 я ведь могу применить 3 разных разложения Шеннона? А ещё я могу менять местами x и -x компоненты?

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_variables', )
    args = parser.parse_args()

    num_variables = args.num_variables

    input_dir = f"res/res_n_{num_variables}"
    input_monom_index_path = os.path.join(input_dir, "monom_id2str.txt")
    input_value_vector2minimum_poly_path = os.path.join(input_dir, "output_value_vector2monom_ids_path.txt")

    val_vector_str2poly_monom_ids = read_value_vector2minimum_poly(input_value_vector2minimum_poly_path)
    monom_id2monom_mask = read_monom_index(input_monom_index_path)

    longest_val_vector_str2poly_monom_ids = filter_longest_polynoms(val_vector_str2poly_monom_ids)
    longest_poly_val_vector_strs = []
    longest_poly_monom_ids = []
    for val_vec_str, monom_ids in longest_val_vector_str2poly_monom_ids.items():
        longest_poly_val_vector_strs.append(val_vec_str)
        longest_poly_monom_ids.append(monom_ids)

    for ids_pair in itertools.combinations(range(len(longest_poly_monom_ids)), 2):
        (i, j) = ids_pair
        assert i != j

        old_monom_masks_1 = longest_poly_monom_ids[i]
        old_monom_masks_2 = longest_poly_monom_ids[j]

        new_monom_masks_1 = [(1,) + m_mask for m_mask in old_monom_masks_1]
        new_monom_masks_2 = [(0,) + m_mask for m_mask in old_monom_masks_2]
        new_merged_monom_masks = new_monom_masks_1 + new_monom_masks_2
        new_merged_monom_mask_strs = set((''.join(str(t) for t in new_merged_monom_masks)))

        new_merged_monom_masks_filtered = []
        for s in new_merged_monom_mask_strs:
            lst = []
            for digit in s:
                lst.append(int(digit))
            new_merged_monom_masks_filtered.append(tuple(lst))
        # Итак, на этом шаге имею новый полином, очищенный от дублирующихся мономов, что дальше?


    # longest_val_vector_numpy2poly_monom_masks = {}
    # for val_vec_str, m_ids in longest_val_vector_str2poly_monom_ids.items():
    #     val_vec_numpy = np.array([])
    #     monoms_list


if __name__ == '__main__':
    main()
