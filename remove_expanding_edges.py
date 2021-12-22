import codecs
import os.path
import re
from queue import PriorityQueue
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from generalized_boolean_polynoms.create_polynoms_graph import Polynom
from generalized_boolean_polynoms.dijkstra_shortest_path import monom_str_to_tuple
from generalized_boolean_polynoms.utils import TRANSFORMATIONS_VERBOSE, TRANSFORMATIONS_VERBOSE_MASKS, LITERALS, \
    monom_mask_to_str, monom_mask_to_tex_str, TRANSFORMATIONS_VERBOSE_TEX_MASKS, polynom_str_to_tex, split_polynom_str, \
    get_polynom_length_from_str, polynom_str_to_monoms_list, polynom_cyclic_shift, is_edge_expanding


def main():
    num_literals = 3
    node_index_path = f"../results/n_{num_literals}/node_index.tsv"
    edges_path = f"../results/n_{num_literals}/edges.tsv"
    output_filtered_edges_path = f"../results/n_{num_literals}/filtered_edges/filt_edges.tsv"
    output_paths = (
        output_filtered_edges_path,
    )
    for out_path in output_paths:
        output_dir = os.path.dirname(out_path)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)

    node_index_df = pd.read_csv(node_index_path, sep='\t', header=None, dtype={"node_id": int, "node_verbose": str},
                                names=["node_id", "node_verbose", "polynom_monoms_str"])
    node_index_df["node_poly_length"] = node_index_df["node_verbose"].apply(get_polynom_length_from_str)
    # node_index_df["polynom_monoms"] = node_index_df["polynom_monoms_str"].apply(polynom_str_to_monoms_list)

    # node_index = node_index_df.set_index("node_id", )["node_verbose"]
    node_id_to_poly_length = node_index_df.set_index("node_id", )["node_poly_length"]
    print("Индекс вершин загружен")
    edges_df = pd.read_csv(edges_path, sep='\t', header=None,
                           names=["poly_1_id", "poly_2_id", "transform_type_id", "literal_id", "target_monom_mask"])
    # edges_df["target_monom_mask"] = edges_df["target_monom_mask"].apply(monom_str_to_tuple)
    print("Рёбра загружены")
    edges_df["is_expanding"] = edges_df.apply(lambda row: is_edge_expanding(row, node_id_to_poly_length), axis=1)
    edges_df = edges_df[~edges_df["is_expanding"]]
    print(f"Edges after filtering: {edges_df.shape[0]}")
    edges_df.drop(columns=["is_expanding"], inplace=True)
    edges_df.to_csv(output_filtered_edges_path, sep='\t', header=None, index=False)


if __name__ == '__main__':
    main()
