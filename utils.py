import codecs
from typing import List, Tuple, Dict, Set
import numpy as np

LITERALS = ['x', 'y', 'z', 't', 'u', 'v']
TRANSFORMATIONS_VERBOSE = {
    1: "x = -x + 1",
    2: "0 = x +(-x) + 1",
    0: "1 = x + -x",
    -1: "-x = x + 1"
}
TRANSFORMATIONS_VERBOSE_MASKS = {
    1: "<literal> = -<literal> + 1",
    2: "0 = <literal> +(-<literal>) + 1",
    0: "1 = <literal> + -<literal>",
    -1: "-<literal> = <literal> + 1"
}

TRANSFORMATIONS_VERBOSE_TEX_MASKS = {
    1: r"<literal> = \bar{<literal>} + 1",
    2: r"0 = <literal> + \bar{<literal>} + 1",
    0: r"1 = <literal> + \bar{<literal>}",
    -1: r"\bar{<literal>} = <literal> + 1"
}


def monom_mask_to_str(monom_mask: List[int]):
    if monom_mask == -1:
        return "0"
    else:
        s = ""
        for literal_id, literal_mask_val in enumerate(monom_mask):
            literal = LITERALS[literal_id]
            if literal_mask_val == 1:
                s += literal
            elif literal_mask_val == 0:
                s += ""
            elif literal_mask_val == -1:
                s += f"(-{literal})"
            else:
                raise ValueError(f"Invalid mono mask value: {literal_mask_val}")
        if s == "":
            s = "1"
        return s


def monom_mask_to_tex_str(monom_mask: List[int]):
    if monom_mask == -1:
        return "0"
    else:
        s = ""
        for literal_id, literal_mask_val in enumerate(monom_mask):
            literal = LITERALS[literal_id]
            if literal_mask_val == 1:
                s += literal
            elif literal_mask_val == 0:
                s += ""
            elif literal_mask_val == -1:
                # s += fr"$\bar{'{' + literal + '}'}$"
                s += r'\bar{' + literal + '}'
            else:
                raise ValueError(f"Invalid mono mask value: {literal_mask_val}")
        if s == "":
            s = "1"
        return s


def polynom_str_to_tex(polynom_str: str):
    for literal in LITERALS:
        polynom_str = polynom_str.replace(f"(-{literal})", r'\bar{' + literal + '}')
        # fr"\bar{'{' + literal + '}'}}")
    return polynom_str


def split_polynom_str(polynom_str: str, ):
    monoms = polynom_str.split('+')
    num_monoms = len(monoms)
    first_half_str = '+'.join(monoms[:num_monoms // 2])
    second_half_str = '+'.join(monoms[num_monoms // 2:])

    return first_half_str, second_half_str


def get_polynom_length_from_str(polynom_str: str) -> int:
    if polynom_str == '0':
        return 0
    else:
        monoms = polynom_str.split('+')
        return len(monoms)


def polynom_monoms_list_to_str(monoms_list: List[Tuple[int]]) -> str:
    monom_strs_list = []
    for monom_tuple in monoms_list:
        monom_str = ','.join((str(x) for x in monom_tuple))
        monom_strs_list.append(monom_str)
    polynom_str = '~~'.join(monom_strs_list)
    return polynom_str


def polynom_str_to_monoms_list(polynom_str: str, monoms_sep="~~", mask_sep=","):
    if polynom_str is np.nan:
        return list()
    else:
        monoms_strs = polynom_str.split(monoms_sep)
        monoms_list = []
        for monom_s in monoms_strs:
            mask_values = [int(x) for x in monom_s.split(mask_sep)]
            monoms_list.append(mask_values)
        monoms_list.sort()
        return monoms_list


def polynom_cyclic_shift(polynom_monoms_list: List[List[int]], n):
    new_polynom_monoms = []
    for monom in polynom_monoms_list:
        np_monom = np.array(monom)
        new_monom = list(np.roll(np_monom, n))
        new_polynom_monoms.append(new_monom)
    new_polynom_monoms.sort()
    return new_polynom_monoms


def is_edge_expanding(row, node_id_to_poly_length) -> bool:
    poly_source_id = row["poly_1_id"]
    poly_target_id = row["poly_2_id"]
    node_source_poly_length = node_id_to_poly_length[poly_source_id]
    node_target_poly_length = node_id_to_poly_length[poly_target_id]
    if node_target_poly_length >= node_source_poly_length:
        return True
    return False


def save_paths_dict(source_id_path_tuples: List[Tuple[int, List[int]]], save_path):
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file:
        for (source_node_id, path) in source_id_path_tuples:
            path_str = ','.join((str(x) for x in path))
            out_file.write(f"{source_node_id}\t{path_str}\n")


def get_monoms_literal_val_monom_literal_id(monoms: List[Tuple[int]], search_lit_val):
    res_monom_id, res_literal_id = -1, -1
    for monom_id, monom in enumerate(monoms):
        for literal_id, literal_value in enumerate(monom):
            if literal_value == search_lit_val:
                return monom_id, literal_id

    return res_monom_id, res_literal_id


def filter_reversed_paths(paths: List[List[int]], keep_ids_set: Set) -> List[Tuple[int, List[int]]]:
    # print(reversed_paths)
    # max_length = max((len(path) for path in reversed_paths))
    kept_paths = []
    # with codecs.open(save_path, 'w+', encoding="utf-8") as out_file, \
    #         codecs.open(save_path_tex, 'w+', encoding="utf-8") as out_file_tex:
    for (vertex_id, reversed_path) in enumerate(paths):
        if vertex_id in keep_ids_set:
            kept_paths.append((vertex_id, reversed_path))
            # longest_paths[vertex_id] = reversed_path
            # s, s_tex = create_polynom_text_and_tex_strings(node_index, vertex_id, reversed_path, edges_df)
            # out_file.write(f"{s}\n")
            # out_file_tex.write(f"{s_tex}\n")
            # longest_paths.append(longest_paths)
    return kept_paths


def print_paths(reversed_paths: List[List[int]], edges_df, node_index):
    """
    ?????????? ???????????????? ???? ???????? ???????????? ?????????????????????????????? ?????????? ???? ?????????????? ????????????, ?????????????????????? ???? ????????????????,
    ?????????? ???????????????? ???????????? ???????? ?? ?????????????? ?????? ????????. ???????? - ?????? ???????????????????????????????????? ?????????????? ????????????, ??????????
    ?????????????? ???????????????? ????????
    :param reversed_paths: ???????????? ?????????????????????????????? ??????????
    """
    for vertex_id, reversed_path in enumerate(reversed_paths):
        s = f"???????? ??????????: {len(reversed_path)}:"
        for i in range(len(reversed_path) - 1):
            polynom_source_id = reversed_path[i]
            polynom_dest_id = reversed_path[i + 1]
            transform_type_id = edges_df[
                (edges_df["poly_1_id"] == polynom_source_id) & (edges_df["poly_2_id"] == polynom_dest_id)][
                "transform_type_id"].values[0]
            transformation_type_verbose = TRANSFORMATIONS_VERBOSE[transform_type_id]
            polynom_source_verbose = node_index[polynom_source_id]
            s += f"[{polynom_source_verbose}] - ({transformation_type_verbose}) -> "
        if len(reversed_path) > 1:
            polynom_dest_verbose = node_index[polynom_dest_id]
            s += polynom_dest_verbose
        print(s)