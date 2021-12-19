from typing import List, Tuple
import numpy as np

LITERALS = ['x', 'y', 'z']
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
    # if len(polynom_monoms_list) > 0:
    new_polynom_monoms = []
    for monom in polynom_monoms_list:
        np_monom = np.array(monom)
        new_monom = list(np.roll(np_monom, n))
        new_polynom_monoms.append(new_monom)
    new_polynom_monoms.sort()
    return new_polynom_monoms

    # num_literals = len(polynom_monoms_list[0])
    #
    # new_array = polynom_monoms_list.copy()
    # for i in range(1, num_literals):
    #     for monom_mask in new_array:
    #         np_monom_mask = np.r
