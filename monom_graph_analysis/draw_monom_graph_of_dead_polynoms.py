import itertools
import os.path

import matplotlib
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from generalized_boolean_polynoms.transform import apply_transformation_to_monom
from generalized_boolean_polynoms.utils import monom_tuple2str, polynom_str_to_monoms_list, monom_tuple2tex_str


def main():
    num_literals = 3
    input_dead_polynoms_path = f"../../results/n_{num_literals}/dead_polynoms/dead_polynoms.tsv"
    save_graphs_dir = f"../../results/n_{num_literals}/dead_polynoms/dead_polynoms_monoms_graphs/"
    all_possible_monom_masks = list(itertools.product((-1, 0, 1), repeat=num_literals))

    if not os.path.exists(save_graphs_dir) and save_graphs_dir != '':
        os.makedirs(save_graphs_dir)

    dead_polynoms_df = pd.read_csv(input_dead_polynoms_path, sep='\t', dtype={"node_id": int, "node_verbose": str})
    for _, row in tqdm(dead_polynoms_df.iterrows(), miniters=dead_polynoms_df.shape[0] // 100,
                       total=dead_polynoms_df.shape[0]):
        dead_polynom_monoms = [tuple(t) for t in polynom_str_to_monoms_list(row["polynom_monoms_str"])]

        polynom_id = row["node_id"]
        graph = nx.Graph()
        graph_save_path = os.path.join(save_graphs_dir, f"{polynom_id}.pdf")
        colors = {}
        for monom_m in all_possible_monom_masks:
            color = "red" if monom_m in dead_polynom_monoms else "blue"

            source_monom_str = monom_tuple2tex_str(monom_m)
            colors[source_monom_str] = color
            graph.add_node(source_monom_str,)
            if color == "red":
                for literal_id in range(num_literals):
                    new_monoms, transform_type = apply_transformation_to_monom(monom_m, literal_id, )
                    for new_m in new_monoms:
                        target_monom_str = monom_tuple2tex_str(new_m)
                        graph.add_edge(source_monom_str, target_monom_str)
        plt.figure(figsize=(7, 7))
        pos = nx.shell_layout(graph)

        color_seq = [colors.get(node) for node in graph.nodes()]
        nx.draw(graph, with_labels=True, font_size=28, pos=pos, node_color=color_seq)
        plt.savefig(graph_save_path, )


if __name__ == '__main__':
    main()
