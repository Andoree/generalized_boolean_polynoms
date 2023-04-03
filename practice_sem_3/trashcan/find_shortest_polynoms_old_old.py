import codecs
import os.path
from collections import Counter
from itertools import product, permutations
from typing import Dict, List

import numpy as np
from queue import Queue

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
    def __init__(self, perm_np_array: np.array, orig_input_sets, ):
        self.perm_np_array = perm_np_array
        # TODO: Мне надо понять, какие значения переменных переходят в какие?
        self.new_input_sets = orig_input_sets[:, perm_np_array]
        orig_input_sets_str = [''.join((str(x) for x in t)) for t in orig_input_sets]
        orig_input_set_str2set_id = {set_str: i for i, set_str in enumerate(orig_input_sets_str)}
        new_input_sets_strs_list = [''.join((str(x) for x in t)) for t in self.new_input_sets]
        self.permuted_val_vec_index = np.array([orig_input_set_str2set_id[set_str] for set_str in new_input_sets_strs_list])
        # print(f"self.perm_np_array{self.perm_np_array}")
        # print(f"self.new_input_sets\n{self.new_input_sets}")
        # print(f"orig_input_sets_str\n{orig_input_sets_str}")
        # print(f"orig_input_set_str2set_id\n{orig_input_set_str2set_id}")
        # print(f"new_input_sets_strs_list\n{new_input_sets_strs_list}")
        # print(self.new_val_vec_index, '\n----')


def create_var_set2monoms_dict(num_vars, input_sets) -> Dict[str, List[Monom]]:
    # 0 - отрицание, 1 - переменная, 2 - константа
    var_set2monoms = {}
    monom_mask_values = (0, 1, 2)
    e_2 = (0, 1)
    for i, monom_tuple in enumerate(product(monom_mask_values, repeat=num_vars)):
        monom_mask_str = ''.join((str(x) for x in monom_tuple))
        monom_positive_mask_str = monom_mask_str.replace("0", "1")
        # print(i, monom_tuple, monom_mask_str, monom_positive_mask_str)
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
    for var_set, monoms in var_set2monoms.items():
        for mon in monoms:
            mon_str = mon.monom_str
            mon_vector = mon.func_vector

            monom_strings.append(mon_str)
            monom_value_vectors.append(mon_vector)

    monom_value_vectors_matrix = np.stack(monom_value_vectors)

    monom_str2id = {m_str: i for i, m_str in enumerate(monom_strings)}

    return monom_str2id, monom_value_vectors_matrix


def main():
    num_vars = 5
    output_dir = f"res_n_{num_vars}"
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    output_monom_id2str_path = os.path.join(output_dir, "monom_id2str.txt")
    output_value_vector_str2monom_ids_path = os.path.join(output_dir, "output_value_vector2monom_ids_path.txt")
    output_value_vector_str2monom_strs_path = os.path.join(output_dir, "output_value_vector_str2monom_strs.txt")
    output_length_stats_path = os.path.join(output_dir, "output_length_stats.txt")


    # Множество двоичных наборов длины n
    input_sets = list(product((0, 1), repeat=num_vars))
    # print("type(input_sets)", type(input_sets))
    # print("type(input_sets[0])", type(input_sets[0]))

    monom_positive_mask2monoms = create_var_set2monoms_dict(num_vars=num_vars, input_sets=input_sets)
    # print("GGGGGGGGGG var_set2monoms", monom_positive_mask2monoms)
    monom2var_set = {}
    for vs, m_list in monom_positive_mask2monoms.items():
        for mnm in m_list:
            monom2var_set[mnm.monom_str] = vs
    # monom2var_set = {: vs for vs, m_list in var_set2monoms.items()}
    var_positive_sets_list = list(monom_positive_mask2monoms.keys())

    # for k, v in monom_positive_mask2monoms.items():
    #     print(f"AA {k} {type(v)} {type(v[0])} {v}")
    # for k, v in monom2var_set.items():
    #     print(f"--------- {k} {v}")
    monom_str2id, monom_value_vectors_matrix = create_monom2id(monom_positive_mask2monoms, num_vars)
    # print("monom_str2id AAAAAAAAA", monom_str2id)
    # for i, (monom_str, monom_id) in enumerate(monom_str2id.items()):
    #     print(f"-> {monom_str} |||| {monom_id} |||| {monom_value_vectors_matrix[i]}")

    with codecs.open(output_monom_id2str_path, 'w+', encoding="utf-8") as out_file:
        for monom_str, monom_id in monom_str2id.items():
            out_file.write(f"{monom_id}\t{monom_str}\n")
    # with codecs.open(output_value_vector_str2monom_ids_path, 'w+', encoding="utf-8") as out_file:
    #     for val_vector_str, monom_ids_list in poly_value_vector_str2monom_ids.items():
    #         out_file.write(f"{val_vector_str}\t{','.join((str(x) for x in monom_ids_list))}\n")

    # with codecs.open(output_value_vector_str2monom_strs_path, 'w+', encoding="utf-8") as out_file:
    #     for val_vector_str, monom_ids_list in poly_value_vector_str2monom_ids.items():
    #         monoms = [monom_id2str[m_id] for m_id in monom_ids_list]
    #         out_file.write(f"{val_vector_str}\t{' + '.join(monoms)}\n")

    # poly_value_vector_str2monom_ids: Dict[str, List[int]] = {}
    found_poly_str_value_vectors = set()
    monoms_queue = Queue()

    p_list = list(permutations(range(num_vars)))
    perms: List[VariablesPermutation] = []
    input_sets = np.array(input_sets)
    for i, p in enumerate(p_list):
        var_perm = VariablesPermutation(perm_np_array=np.array(p), orig_input_sets=input_sets)
        perms.append(var_perm)

    num_found_functions = 0
    with codecs.open(output_value_vector_str2monom_ids_path, 'w+', encoding="utf-8") as value_vector_str2monom_ids_file:
        for var_set in var_positive_sets_list:
            monoms = monom_positive_mask2monoms[var_set]
            for m in monoms:
                monom_string = m.monom_str
                new_monom_mask = m.monom_mask
                monom_id = monom_str2id[monom_string]
                new_monom_func_vector = monom_value_vectors_matrix[monom_id]
                new_monom_func_vector_str = ''.join((str(x) for x in new_monom_func_vector))

                if new_monom_func_vector_str in found_poly_str_value_vectors:
                    continue

                added_monom_ids = {monom_id, }
                added_monom_masks = {new_monom_mask, }
                added_monoms_variables_positive_mask = {var_set}
                print("var_set", var_set)
                print("new_monom_mask", new_monom_mask)
                print("added_monom_masks", added_monom_masks)
                # print(">>>>>>>", added_monom_masks)
                # print(f">>>>> type(added_monom_ids) {type(added_monom_ids)}")
                # print(f">>>>> type(added_monom_masks) {type(added_monom_masks)}")
                # print(f">>>>> type(added_monoms_variables_positive_mask) {type(added_monoms_variables_positive_mask)}")
                # print(f">>>>> type(new_monom_func_vector) {type(new_monom_func_vector)}")
                # print('--')
                monoms_queue.put(
                    (added_monom_ids, added_monom_masks, added_monoms_variables_positive_mask, new_monom_func_vector))
                # poly_value_vector_str2monom_ids[new_monom_func_vector_str] = list(added_monom_ids)
                # found_poly_str_value_vectors.add(new_monom_func_vector_str)
                for perm in perms:
                    perm_np_array = perm.perm_np_array
                    permuted_val_vec_index = perm.permuted_val_vec_index
                    permuted_poly_func_vector = new_monom_func_vector[permuted_val_vec_index]
                    permuted_poly_func_vector_str = ''.join((str(x) for x in permuted_poly_func_vector))
                    if permuted_poly_func_vector_str not in found_poly_str_value_vectors:
                        found_poly_str_value_vectors.add(permuted_poly_func_vector_str)

                        # poly_value_vector_str2monom_ids[new_poly_func_vector_str] = new_monom_ids_set

                        num_found_functions += 1
                        value_vector_str2monom_ids_file \
                            .write(f"{permuted_poly_func_vector_str}\t{','.join((str(x) for x in added_monom_ids))}"
                                   f"\t{','.join((str(x) for x in perm_np_array))}\n")


    print("NUM FOUND, MONOMS", num_found_functions)
    j = 0
    for var_set in var_positive_sets_list:
        monoms = monom_positive_mask2monoms[var_set]
        for m in monoms:
            j += 1
            # print(j, "M", m)
    # print("KKKKKKKKKKK", found_poly_str_value_vectors)
    expected_num_functions = int(2 ** (2 ** num_vars)) - 1
    print(f"Expected num function: {expected_num_functions}")

    # added_monom_strs_list = set()
    zero_value_vector = np.zeros(shape=(int((2 ** num_vars))), dtype=np.int)
    with codecs.open(output_value_vector_str2monom_ids_path, 'a+', encoding="utf-8") as value_vector_str2monom_ids_file:
        # for val_vector_str, monom_ids_list in poly_value_vector_str2monom_ids.items():
        #     out_file.write(f"{val_vector_str}\t{','.join((str(x) for x in monom_ids_list))}\n")
        while len(found_poly_str_value_vectors) != expected_num_functions:
            # print(num_found_functions)
            # print(" len(found_poly_str_value_vectors)", len(found_poly_str_value_vectors))

            (current_monom_ids, current_monom_masks, current_monoms_positive_masks, current_polynom_func_vector) = \
                monoms_queue.get()
            current_polynom_func_vector_str = ''.join((str(x) for x in current_polynom_func_vector))
            # if current_polynom_func_vector_str in found_poly_str_value_vectors:
            #     continue
            # Обходим возможных кандидатов на добавление в список мономов
            # Сначала обходим маски (var_set), содержащие только отрицания
            for new_monom_positive_mask, monoms_list in monom_positive_mask2monoms.items():
                if new_monom_positive_mask in current_monoms_positive_masks:
                    continue
                for monom in monoms_list:
                    monom_str = monom.monom_str
                    new_monom_mask = monom.monom_mask
                    new_monom_func_vector = monom.func_vector

                    new_poly_func_vector = (current_polynom_func_vector + new_monom_func_vector) % 2
                    new_poly_func_vector_str = ''.join((str(x) for x in new_poly_func_vector))

                    # if poly_value_vector_str2monom_ids.get(new_poly_func_vector_str) is not None:
                    #     continue
                    if new_poly_func_vector_str in found_poly_str_value_vectors:
                        continue
                    else:
                        new_monom_id = monom_str2id[monom_str]
                        # new_monom_masks = current_monom_masks.copy()

                        # new_monom_ids_set = current_monom_ids + [new_monom_id, ]
                        assert new_monom_id not in current_monom_ids
                        new_monom_ids_set = current_monom_ids.copy()
                        new_monom_ids_set.add(new_monom_id)
                        # new_monom_masks = current_monom_masks + [monom_mask, ]
                        assert new_monom_mask not in current_monom_masks
                        new_monom_masks = current_monom_masks.copy()
                        new_monom_masks.add(new_monom_mask)

                        # new_monom_positive_masks = current_monoms_positive_masks + [new_monom_positive_mask, ]
                        assert new_monom_positive_mask not in current_monoms_positive_masks
                        new_monom_positive_masks = current_monoms_positive_masks.copy()
                        new_monom_positive_masks.add(new_monom_mask)
                        if new_poly_func_vector_str not in found_poly_str_value_vectors:
                            monoms_queue.put(
                                (new_monom_ids_set, new_monom_masks, new_monom_positive_masks, new_poly_func_vector))

                        for perm in perms:
                            perm_np_array = perm.perm_np_array
                            permuted_val_vec_index = perm.permuted_val_vec_index
                            permuted_poly_func_vector = new_poly_func_vector[permuted_val_vec_index]
                            permuted_poly_func_vector_str = ''.join((str(x) for x in permuted_poly_func_vector))
                            if permuted_poly_func_vector_str not in found_poly_str_value_vectors:
                                found_poly_str_value_vectors.add(permuted_poly_func_vector_str)
                                # poly_value_vector_str2monom_ids[new_poly_func_vector_str] = new_monom_ids_set
                                if len(found_poly_str_value_vectors) % 100000 == 0:
                                    print(f"{len(found_poly_str_value_vectors) // 1000} / {expected_num_functions // 1000}", )
                                num_found_functions += 1
                                value_vector_str2monom_ids_file\
                                    .write(f"{permuted_poly_func_vector_str}\t{','.join((str(x) for x in new_monom_ids_set))}"
                                           f"\t{','.join((str(x) for x in perm_np_array))}\n")

                    # (current_monom_ids, current_monom_masks, current_monoms_positive_masks, current_polynom_func_vector)

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
