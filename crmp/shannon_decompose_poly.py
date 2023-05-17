import argparse
import codecs
import itertools
import os
from typing import Dict, Tuple, List

import numpy as np


def read_monom_index(inp_path: str) -> Dict[int, Tuple[int]]:
    monom_id2monom_mask = {}
    monom_id2str = {}
    with codecs.open(inp_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            monom_id = int(attrs[0])
            monom_mask = tuple(int(x) for x in attrs[1].split(','))
            monom_str = attrs[2]

            monom_id2monom_mask[monom_id] = monom_mask
            monom_id2str[monom_id] = monom_str
    return monom_id2monom_mask, monom_id2str


def read_value_vector2minimum_poly(inp_path) -> Dict[str, Tuple[int]]:
    val_vector_str2monom_ids = {}
    with codecs.open(inp_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            val_vector_str = attrs[0]
            if len(attrs) > 1:
                monom_ids = tuple(int(x) for x in attrs[1].split(','))
            else:
                monom_ids = tuple()
            # if len(attrs[1].split(','))
            val_vector_str2monom_ids[val_vector_str] = monom_ids
    return val_vector_str2monom_ids


def filter_longest_polynoms(val_vector_str2poly_monom_ids: Dict[str, Tuple[int]]) -> Dict[str, Tuple[int]]:
    maximum_length = max((len(t) for t in val_vector_str2poly_monom_ids.values()))
    longest_val_vector_str2poly_monom_ids = \
        {key: t for key, t in val_vector_str2poly_monom_ids.items() if len(t) == maximum_length}

    return longest_val_vector_str2poly_monom_ids




def load_results_grouped_by_num_variables(input_dir, num_vars_start, num_vars_end):
    res_dict = {}
    for num_vars in range(num_vars_start, num_vars_end + 1):
        res_dict[num_vars] = {}
        input_subdir = os.path.join(input_dir, f"res_n_{num_vars}")

        monom_id2str_path = os.path.join(input_subdir, "monom_id2str.txt")
        value_vector2monom_ids_path = os.path.join(input_subdir, "output_value_vector2monom_ids_path.txt")
        # value_vector_str2monom_strs = os.path.join(input_subdir, "output_value_vector_str2monom_strs.txt")

        monom_id2monom_mask, monom_id2str  = read_monom_index(monom_id2str_path)
        val_vector_str2monom_ids = read_value_vector2minimum_poly(value_vector2monom_ids_path)

        res_dict[num_vars]["monom_id2monom_mask"] = monom_id2monom_mask
        res_dict[num_vars]["monom_id2str"] = monom_id2str
        print("AAAAAAAAA monom_id2str", monom_id2str)
        res_dict[num_vars]["val_vector_str2monom_ids"] = val_vector_str2monom_ids

    return res_dict


def calculate_single_monom_value_vector(monom_mask: Tuple[int], inp_sets: List[Tuple[int]]) -> np.array:
    values = []
    monom_mask = tuple(x for x in monom_mask if x != -1)
    for in_set in inp_sets:
        zero_value = False
        for i in range(len(in_set)):
            variable_value = in_set[i]
            mask_value = monom_mask[i]
            # variable_value = 0 & mask_value = 0 (-x) -> 1
            # variable_value = 0 & mask_value = 1 (x) -> 0
            # variable_value = 1 & mask_value = 0 (-x) -> 0
            # variable_value = 1 & mask_value = 1 (x) -> 1
            if (variable_value != mask_value) and mask_value != 2:
                values.append(0)
                zero_value = True
                break
        if not zero_value:
            values.append(1)
    assert len(values) == len(inp_sets)
    return np.array(values)


def calculate_monom_list_poly_value_vector(monom_masks_list, inp_sets) -> np.array:
    if len(monom_masks_list) == 0:
        return np.zeros(shape=len(inp_sets), dtype=np.int)
    monom_value_vectors: List[np.array] = []
    for monom_mask in monom_masks_list:
        val_vec_np_array = calculate_single_monom_value_vector(monom_mask, inp_sets)
        monom_value_vectors.append(val_vec_np_array)
    monom_value_vectors = np.array(monom_value_vectors)

    value_vec = monom_value_vectors.sum(axis=0) % 2

    return value_vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_variables', )
    args = parser.parse_args()

    num_variables = args.num_variables
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
    polynoms_dict = load_results_grouped_by_num_variables('./', num_vars_start=num_variables - 1, num_vars_end=num_variables)

    input_dir_to_decompose = f"res_n_{num_variables}"
    input_monom_index_to_decompose_path = os.path.join(input_dir_to_decompose, "monom_id2str.txt")
    input_value_vector2minimum_poly_to_decompose_path = os.path.join(input_dir_to_decompose,
                                                                     "output_value_vector2monom_ids_path.txt")


    val_vector_str2poly_monom_ids = read_value_vector2minimum_poly(input_value_vector2minimum_poly_to_decompose_path)
    monom_id2monom_mask, monom_id2str = read_monom_index(input_monom_index_to_decompose_path)

    longest_val_vector_str2poly_monom_ids = filter_longest_polynoms(val_vector_str2poly_monom_ids)
    longest_poly_val_vector_strs: List[str] = []
    longest_poly_monom_ids: List[Tuple[int]] = []
    longest_poly_monom_masks: List[List[Tuple[int]]] = []


    output_dir = f"./res_n_{num_variables}/"
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_decomposition_file_path = os.path.join(output_dir, "poly_decomposition.txt")

    for val_vec_str, monom_ids in longest_val_vector_str2poly_monom_ids.items():
        longest_poly_val_vector_strs.append(val_vec_str)
        longest_poly_monom_ids.append(monom_ids)
        monom_masks = [monom_id2monom_mask[m_id] for m_id in monom_ids]
        longest_poly_monom_masks.append(monom_masks)

    # list_of_existing_x_monom_masks: List[List[Tuple[int]]] = []
    # list_of_non_x_monom_masks: List[List[Tuple[int]]] = []
    # list_of_no_literal_x_masks: List[List[Tuple[int]]] = []
    with codecs.open(output_decomposition_file_path, "w+", encoding="utf-8") as out_file:
        for longest_poly_monom_ids, longest_monom_masks_list in zip(longest_poly_monom_ids, longest_poly_monom_masks):
            """
            Далее: 
            1. Мне нужно написать унифицированную подгрузку мономов меньшей длины. 
            2. Потом мне нужно что? Может быть, перебирать все переменные, по которым будет происходить разложение
            3. Далее я получаю 3 полинома длины не более n - 1, но при этом такое, что k + l + m = n
            """
            for variable_id in range(num_variables):
                # Тут будет происходить разложение:
                existing_x_monom_masks = []
                non_x_monom_masks = []
                no_literal_x_masks = []
                for monom_mask_tuple in longest_monom_masks_list:
                    variable_value = monom_mask_tuple[variable_id]
                    new_monom_mask_tuple = [0, ] * num_variables  # (num_variables - 1)
                    for monom_var_id in range(num_variables):
                        if monom_var_id != variable_id:
                            # print("new_monom_mask_tuple", new_monom_mask_tuple, monom_var_id)
                            # print("monom_mask_tuple", monom_mask_tuple, monom_var_id)
                            new_monom_mask_tuple[monom_var_id] = monom_mask_tuple[monom_var_id]
                        else:
                            new_monom_mask_tuple[monom_var_id] = -1
                    new_monom_mask_tuple = tuple(new_monom_mask_tuple)
                    # Сконструировал новую маску, теперь выясняю, в какую подфункцию она уйдёт

                    if variable_value == 0:
                        non_x_monom_masks.append(new_monom_mask_tuple)
                    if variable_value == 1:
                        existing_x_monom_masks.append(new_monom_mask_tuple)
                    if variable_value == 2:
                        no_literal_x_masks.append(new_monom_mask_tuple)

                # Тут у меня 3 функции. Теперь для каждой из функций мне надо узнать вектор значений, который она задаёт
                # По вектору значений я пойму что? Надо ассертнуть, что в списке минимальных полином именно такой длины есть
                # Далее я должен в файл записать:
                # <исходный полином> <по какой перем.разложили> <k,l,m> <3 полинома от n-1 переменной>

                # Здесь сразу обработать
                input_sets_less_vars = list(itertools.product((0, 1), repeat=num_variables - 1))
                existing_x_poly_val_vec = calculate_monom_list_poly_value_vector(existing_x_monom_masks,
                                                                                 inp_sets=input_sets_less_vars)
                non_x_poly_val_vec = calculate_monom_list_poly_value_vector(non_x_monom_masks,
                                                                            inp_sets=input_sets_less_vars)
                no_literal_x_poly_val_vec = calculate_monom_list_poly_value_vector(no_literal_x_masks,
                                                                                   inp_sets=input_sets_less_vars)

                existing_x_poly_val_vec_str = ''.join((str(x) for x in existing_x_poly_val_vec))
                # for k, v in polynoms_dict[num_variables - 1].items():
                #     print(">>>>>>>> k", k)
                #     print(">>>>>>>> v", v)
                existing_x_poly_monom_ids_list = \
                    polynoms_dict[num_variables - 1]["val_vector_str2monom_ids"][existing_x_poly_val_vec_str]
                existing_x_poly_monom_string_list = \
                    [polynoms_dict[num_variables - 1]["monom_id2str"][m_id] for m_id in existing_x_poly_monom_ids_list]

                non_x_poly_val_vec_str = ''.join((str(x) for x in non_x_poly_val_vec))
                non_x_poly_monom_ids_list = \
                    polynoms_dict[num_variables - 1]["val_vector_str2monom_ids"][non_x_poly_val_vec_str]
                non_x_poly_monom_string_list = \
                    [polynoms_dict[num_variables - 1]["monom_id2str"][m_id] for m_id in non_x_poly_monom_ids_list]

                no_literal_x_poly_val_vec_str = ''.join((str(x) for x in no_literal_x_poly_val_vec))
                no_literal_x_poly_monom_ids_list = \
                    polynoms_dict[num_variables - 1]["val_vector_str2monom_ids"][no_literal_x_poly_val_vec_str]
                no_literal_x_poly_monom_string_list = \
                    [polynoms_dict[num_variables - 1]["monom_id2str"][m_id] for m_id in no_literal_x_poly_monom_ids_list]

                longest_monom_mask_strs_list = \
                    [polynoms_dict[num_variables]["monom_id2str"][m_id] for m_id in longest_poly_monom_ids]



                out_file.write(f"{' + '.join(str(x) for x in longest_monom_mask_strs_list)}\t"
                               f"{variable_id}\t"
                               f"{len(existing_x_poly_monom_string_list)},{len(non_x_poly_monom_string_list)},"
                               f"{len(no_literal_x_poly_monom_string_list)}\t"
                               f"{' + '.join(str(x) for x in existing_x_poly_monom_string_list)}\t"
                               f"{' + '.join(str(x) for x in non_x_poly_monom_string_list)}\t"
                               f"{' + '.join(str(x) for x in no_literal_x_poly_monom_string_list)}\n")




        # list_of_existing_x_monom_masks.append(existing_x_monom_masks)
        # list_of_non_x_monom_masks.append(non_x_monom_masks)
        # list_of_no_literal_x_masks.append(no_literal_x_masks)


if __name__ == '__main__':
    main()
