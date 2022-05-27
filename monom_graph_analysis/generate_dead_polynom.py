import itertools

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from generalized_boolean_polynoms.transform import apply_transformation_to_monom
from generalized_boolean_polynoms.utils import monom_tuple2str


def create_monoms_transformations_adjacency_lists(monom_masks, monom_mask_str2monom_id, num_literals):
    adjacency_lists = [set() for _ in monom_masks]
    for monom_m in monom_masks:
        source_monom_str = monom_tuple2str(monom_m)
        source_monom_id = monom_mask_str2monom_id[source_monom_str]
        for literal_id in range(num_literals):
            new_monoms, transform_type = apply_transformation_to_monom(monom_m, literal_id, )
            for new_m in new_monoms:
                target_monom_str = monom_tuple2str(new_m)
                target_monom_id = monom_mask_str2monom_id[target_monom_str]
                adjacency_lists[source_monom_id].add(target_monom_id)
    return adjacency_lists


def calculate_monom_value(input_set, monom_mask):
    monom_value = 1
    for input_literal_value, monom_literal_value in zip(input_set, monom_mask):
        if monom_literal_value == 0 or (input_literal_value == 1 and monom_literal_value == 1) or \
                (input_literal_value == 0 and monom_literal_value == -1):
            monom_value *= 1
        # (0, 1)  (1 -1)
        elif (input_literal_value == 0 and monom_literal_value == 1) or \
                (input_literal_value == 1 and monom_literal_value == -1):
            monom_value = 0
        else:
            raise ValueError(f"Invalid polynom_multiplication: input value"
                             f"{input_literal_value} and monom mask {monom_literal_value}")
    return monom_value


def check_polynom_equals_values(num_literals, polynom_monoms, output_values):
    input_sets = list(itertools.product((0, 1), repeat=num_literals))
    for inp_s, out_v in zip(input_sets, output_values):
        monoms_sum = 0
        for monom in polynom_monoms:
            monom_value = calculate_monom_value(input_set=inp_s, monom_mask=monom)
            monoms_sum += monom_value
        if monoms_sum % 2 == out_v:
            continue
        else:
            return False
    return True

# TODO
def main():
    num_literals = 3
    # save_graph_path = f"../../results/n_{num_literals}/monoms_graph/monom_transformations_graph.png"
    monom_masks = list(itertools.product((-1, 0, 1), repeat=num_literals))
    monom_mask_str2monom_id = {monom_tuple2str(monom_m): idx for idx, monom_m in enumerate(monom_masks)}
    adjacency_lists = [set() for _ in monom_masks]
    for monom_m in monom_masks:
        source_monom_str = monom_tuple2str(monom_m)
        source_monom_id = monom_mask_str2monom_id[source_monom_str]
        for literal_id in range(num_literals):
            new_monoms, transform_type = apply_transformation_to_monom(monom_m, literal_id, )
            for new_m in new_monoms:
                target_monom_str = monom_tuple2str(new_m)
                target_monom_id = monom_mask_str2monom_id[target_monom_str]
                adjacency_lists[source_monom_id].add(target_monom_id)


if __name__ == '__main__':
    main()
