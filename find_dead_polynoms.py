import codecs
import os
from typing import List

import pandas as pd

from generalized_boolean_polynoms.classes import Polynom
from generalized_boolean_polynoms.transform import check_polynom_has_non_expanding_transform
from generalized_boolean_polynoms.utils import polynom_str_to_monoms_list, polynom_str_to_tex, polynom_cyclic_shift


def filter_polynoms(polynoms: List[Polynom], num_literals) -> List[int]:
    unique_polynoms_string_dict = {}
    for poly_id, poly in enumerate(polynoms):
        met_before = False
        polynom_monoms_list = poly.monoms
        for i in range(1, num_literals):
            cycled_polynom_monoms_list = polynom_cyclic_shift(polynom_monoms_list, n=i)
            poly = Polynom(monoms=cycled_polynom_monoms_list)
            cycled_poly_str = str(poly)
            if unique_polynoms_string_dict.get(cycled_poly_str) is not None:
                met_before = True
                break
        if met_before == False:
            poly = Polynom(polynom_monoms_list)
            poly_str = str(poly)
            unique_polynoms_string_dict[poly_str] = poly_id
    keep_poly_ids = list(unique_polynoms_string_dict.values())
    return keep_poly_ids


def save_dead_polynoms(dead_polynoms_tex_list: List[str], save_path: str):
    with codecs.open(save_path, 'w+', encoding="utf-8") as inp_file:
        for tex_str in dead_polynoms_tex_list:
            s = "\\begin{dmath}\n" + '{' + f"{tex_str}" + '}\n'
            s += "\\end{dmath}\n"
            inp_file.write(s)


def main():
    num_literals = 3
    node_index_path = f"../results/n_{num_literals}/node_index.tsv"
    expanding_paths_with_min_path_nodes_path = f"../results/n_{num_literals}/expanding_paths/filt_exp_paths_nodex.tsv"
    output_dead_polynoms_tex_path = f"../results/n_{num_literals}/expanding_paths/dead_polynoms.tex"
    output_paths = (
        output_dead_polynoms_tex_path,
    )
    for out_path in output_paths:
        output_dir = os.path.dirname(out_path)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)
    node_index_df = pd.read_csv(node_index_path, sep='\t', header=None, dtype={"node_id": int, "node_verbose": str},
                                names=["node_id", "node_verbose", "polynom_monoms_str"])
    exp_paths_polynoms_df = pd.read_csv(expanding_paths_with_min_path_nodes_path, sep='\t', header=None,
                                        dtype={"node_id": int, "min_path_nodes": str},
                                        names=["node_id", "min_path_nodes"])
    merged_df = exp_paths_polynoms_df.merge(node_index_df, how='inner', on="node_id")
    print("dead", exp_paths_polynoms_df.shape)
    print("merged_df", merged_df.shape)
    print(exp_paths_polynoms_df)
    print(merged_df)
    merged_df["poly_object"] = merged_df["polynom_monoms_str"].apply(lambda x: Polynom(polynom_str_to_monoms_list(x)))
    print(merged_df)
    merged_df["has_non_expanding_transform"] = merged_df["poly_object"].apply(
        lambda x: check_polynom_has_non_expanding_transform(poly=x, num_literals=num_literals))
    has_no_expanding_transform_df = merged_df[~merged_df["has_non_expanding_transform"]]
    has_no_expanding_transform_df["polynom_tex"] = has_no_expanding_transform_df["node_verbose"].apply(
        polynom_str_to_tex)
    # dead_polynoms = has_no_expanding_transform_df["poly_object"].values
    # filtered_dead_polynoms_ids = filter_polynoms(polynoms=dead_polynoms, num_literals=num_literals)
    # print("Dead polynoms before filtering:", len(dead_polynoms))
    # print("Dead polynoms after filtering:", len(filtered_dead_polynoms_ids))
    save_dead_polynoms(dead_polynoms_tex_list=has_no_expanding_transform_df["polynom_tex"].values,
                       save_path=output_dead_polynoms_tex_path)
    # print(has_no_expanding_transform_df)
    # print(has_no_expanding_transform_df.shape)


if __name__ == '__main__':
    main()
