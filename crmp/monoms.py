from typing import Dict

import numpy as np


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
