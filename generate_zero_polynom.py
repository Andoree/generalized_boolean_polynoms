import itertools
from queue import PriorityQueue
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from generalized_boolean_polynoms.classes import Polynom
from generalized_boolean_polynoms.transform import apply_transformation_to_monom, \
    check_polynom_has_non_expanding_transform


def generate_A_n_polynom(mask: Tuple[int]) -> Polynom:
    num_variables = len(mask)
    polynoms = []
    for i, mask_val in enumerate(mask):
        if mask_val == 1:
            A_i_mask = [0, ] * num_variables
            A_i_mask[i] = 1
            monoms = [A_i_mask, ]
        elif mask_val == 0:
            A_i_mask = [0, ] * num_variables
            monoms = [A_i_mask, ]
        elif mask_val == -1:
            A_mask_not_x = [0, ] * num_variables
            A_mask_x = [0, ] * num_variables
            A_mask_1 = [0, ] * num_variables
            A_mask_not_x[i] = -1
            A_mask_x[i] = 1
            monoms = [A_mask_not_x, A_mask_x, A_mask_1]
        else:
            raise ValueError(f"Invalid mask value: {mask}")
        poly = Polynom(monoms)
        polynoms.append(poly)
    result_polynom = Polynom([[0, ] * num_variables, ])
    for poly in polynoms:
        result_polynom *= poly
    return result_polynom


def create_integer_to_An_mapping(n) -> Dict[int, Polynom]:
    # В маске должна быть по крайней мере одна минус единица
    possible_values = (-1, 0, 1)
    i = 0
    int_An_mapping = {}
    for mask in itertools.product(possible_values, repeat=n):
        if -1 in mask:
            # int_An_mapping[i] = mask
            poly = generate_A_n_polynom(mask=mask)
            int_An_mapping[i] = poly
            i += 1
    unique_polynoms = set((str(x) for x in int_An_mapping.values()))
    assert len(unique_polynoms) == len(int_An_mapping)
    return int_An_mapping


# TODO:
"""
    1. Составить произведение A_i
    2. 
    Как нумеровать? Ну по сути маска - это троичное число. Может

"""


def generate_zero_polynom(n: int, A_s_dict: [int, Polynom]) -> Polynom:
    num_As = int(3 ** n - 2 ** n)
    presence_mask = np.random.choice([0, 1], size=(num_As,), )
    monoms = []
    for i, polynom_presence_mask_value in enumerate(presence_mask):
        if polynom_presence_mask_value == 1:
            A_polynom = A_s_dict[i]
            A_monoms = A_polynom.monoms
            monoms.extend(A_monoms)
    generated_poly = Polynom(monoms)
    return generated_poly


def traverse_single_polynom(poly: Polynom, depth: int, num_expanding: int,
                            traverse_history: Dict[str, Tuple[int, Polynom]]):
    expanding_candidates = []
    current_polynom_length = len(poly)
    # Если полином уже пуст, прекратить обход
    if current_polynom_length == 0:
        return depth, num_expanding
    for monom_mask in poly.monoms:
        num_literals = len(monom_mask)
        for literal_id in range(num_literals):
            # Заводим копию полинома
            new_polynom_monoms = list(poly.monoms)
            new_polynom_monoms.remove(monom_mask)
            new_monoms, transform_type = apply_transformation_to_monom(monom_mask, literal_id, )
            # Добавляем новые мономы в новый полином
            new_polynom_monoms.extend(new_monoms)
            # Создаём новый полином по результатам преобразования
            new_polynom = Polynom(new_polynom_monoms)
            # Если не обходили полином, то возможно, обходим
            if traverse_history.get(str(new_polynom)) is None:
                # Если новый полином короче, обходим его
                if len(new_polynom) < current_polynom_length:
                    traverse_history[str(new_polynom)] = (depth + 1, poly)
                    recursive_return_depth, recursive_return_num_expanding = traverse_single_polynom(poly=new_polynom,
                                                                                                     depth=depth + 1,
                                                                                                     num_expanding=num_expanding,
                                                                                                     traverse_history=traverse_history)
                    if recursive_return_depth is not None:
                        return recursive_return_depth, recursive_return_num_expanding
                else:
                    expanding_candidates.append(new_polynom)
    for expanding_candidate_polynom in expanding_candidates:
        traverse_history[str(expanding_candidate_polynom)] = (depth + 1, poly)
        recursive_return_depth, recursive_return_num_expanding = traverse_single_polynom(
            poly=expanding_candidate_polynom, depth=depth + 1,
            num_expanding=num_expanding + 1,
            traverse_history=traverse_history)
        if recursive_return_depth is not None:
            return recursive_return_depth, recursive_return_num_expanding
    return None, None


def traverse_polynom_transformations(start_poly: Polynom):
    traverse_history = {}
    depth = traverse_single_polynom(poly=start_poly, depth=0, num_expanding=0, traverse_history=traverse_history)
    return depth
    # polynom_length, current_poly = vertices_to_visit_pq.get()


def monte_carlo_transformations(n: int, num_samples: int):
    A_s_dict = create_integer_to_An_mapping(n=n)
    expanding_stats_dict = {}
    depth_stats_dict = {}
    dead_polynoms_stats_dict = {False: 0, True: 0}
    for i in tqdm(range(num_samples), miniters=num_samples // 100):
        start_poly = generate_zero_polynom(n=n, A_s_dict=A_s_dict)
        is_polynom_not_dead = check_polynom_has_non_expanding_transform(poly=start_poly, num_literals=n)
        if is_polynom_not_dead:
            dead_polynoms_stats_dict[False] += 1
        else:
            dead_polynoms_stats_dict[True] += 1
        depth, num_expanding = traverse_polynom_transformations(start_poly=start_poly)
        # print("Depth:", depth, "Num expanding:", num_expanding, "Length:", len(start_poly))
        if depth_stats_dict.get(depth) is None:
            depth_stats_dict[depth] = 0
        depth_stats_dict[depth] += 1
        if expanding_stats_dict.get(num_expanding) is None:
            expanding_stats_dict[num_expanding] = 0
        expanding_stats_dict[num_expanding] += 1

    return depth_stats_dict, expanding_stats_dict, dead_polynoms_stats_dict


def main():
    i = 0
    # for t in itertools.product(possible_values, repeat=4):
    #     i += 1
    #     print(i, t, type(t))
    # d = create_integer_to_An_mapping(n=4)
    # for k, v in d.items():
    #     print(k, v)
    # print("Длина:", len(d))
    # A_s_dict = create_integer_to_An_mapping(n=4)
    # poly = generate_zero_polynom(n=4, A_s_dict=A_s_dict)
    # print(poly)
    depth_stats_dict, expanding_stats_dict, dead_polynoms_stats_dict = monte_carlo_transformations(n=4,
                                                                                                   num_samples=1000000)
    print("Depth statistics:")
    for k, v in depth_stats_dict.items():
        print(f"{k}: {v}")
    print('-' * 10)
    print("Expanding transformations statistics:")
    for k, v in expanding_stats_dict.items():
        print(f"{k}: {v}")
    print('-' * 10)
    print("Dead polynoms statistics:")
    for k, v in dead_polynoms_stats_dict.items():
        print(f"{k}: {v}")




if __name__ == '__main__':
    main()
