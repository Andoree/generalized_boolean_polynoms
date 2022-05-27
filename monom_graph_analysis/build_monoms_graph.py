import itertools
import os
from typing import List, Tuple, Dict, Set

import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

from generalized_boolean_polynoms.transform import apply_transformation_to_monom
from generalized_boolean_polynoms.utils import monom_tuple2str, monom_tuple2tex_str


def create_monoms_graph_adjacency_lists(num_literals: int) -> Tuple[List[List[int]], Dict[Tuple, int]]:
    monom_masks = list((itertools.product((-1, 0, 1), repeat=num_literals)))
    monom2node_id = {m: i for i, m in enumerate(monom_masks)}

    adjacency_lists: List[List[int]] = [[] for i in range(len(monom_masks))]
    for monom_m in monom_masks:
        source_monom_id = monom2node_id[monom_m]
        for literal_id in range(num_literals):
            new_monoms, transform_type = apply_transformation_to_monom(monom_m, literal_id, )
            new_monoms = [tuple(t) for t in new_monoms]
            for new_m in new_monoms:
                new_monom_id = monom2node_id[new_m]
                adjacency_lists[source_monom_id].append(new_monom_id)
    return adjacency_lists, monom2node_id

# TODO: Перенесено в другой скрипт
# def traverse_monoms_graph(adjacency_lists):
#     num_monoms = len(adjacency_lists)
#     dead_polynoms_pool: List[Set[int]] = []
#     for start_node_id in tqdm(range(num_monoms), total=num_monoms):
#         start_visited_nodes: Set[int] = set()
#         start_excluded_nodes: Set[int] = set()
#
#         start_neighbors = set(adjacency_lists[start_node_id])
#         start_visited_nodes.add(start_node_id)
#         # Исключаем из кандидатов на посещение соседей стартовой вершины и саму стартовую
#         start_excluded_nodes = start_excluded_nodes.union(start_neighbors)
#         start_excluded_nodes.add(start_node_id)
#
#         visit_polynom_node(adjacency_lists=adjacency_lists, visited_nodes=start_visited_nodes,
#                            excluded_nodes=start_excluded_nodes, num_nodes=num_monoms,
#                            dead_polynoms_pool=dead_polynoms_pool)
#     return dead_polynoms_pool
#
# def visit_polynom_node(adjacency_lists, visited_nodes: Set[int], excluded_nodes: Set[int], num_nodes: int,
#                        dead_polynoms_pool: List[Set[int]]):
#     nodes_to_visit = [node_id for node_id in range(num_nodes) if node_id not in excluded_nodes]
#     for next_hop_node_id in nodes_to_visit:
#         # В список уже посещённых полиномов передаёт посещённые ранее + следующую обрабатываемую
#         next_hop_visited_nodes = visited_nodes.copy()
#         next_hop_visited_nodes.add(next_hop_node_id)
#
#         # Исключаем из кандидатов на посещение соседей обрабатываемой вершины
#         next_hop_neighbors = set(adjacency_lists[next_hop_node_id])
#         next_hop_excluded_nodes = excluded_nodes.copy()
#         next_hop_excluded_nodes = next_hop_excluded_nodes.union(next_hop_neighbors)
#         # Исключаем также и обрабатываемую на следующем шаге вершину
#         next_hop_excluded_nodes.add(next_hop_node_id)
#
#         # Рекурсивный обход
#         visit_polynom_node(adjacency_lists=adjacency_lists, visited_nodes=next_hop_visited_nodes,
#                            excluded_nodes=next_hop_excluded_nodes, num_nodes=num_nodes,
#                            dead_polynoms_pool=dead_polynoms_pool)
#     dead_polynoms_pool.append(visited_nodes)


def main():
    num_literals = 2
    save_graph_path = f"../../results/n_{num_literals}/monoms_graph/monom_transformations_graph.pdf"
    output_dir = os.path.dirname(save_graph_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    monom_masks = list(itertools.product((-1, 0, 1), repeat=num_literals))
    graph = nx.Graph()

    for monom_m in monom_masks:
        source_monom_str = monom_tuple2tex_str(monom_m)
        for literal_id in range(num_literals):
            new_monoms, transform_type = apply_transformation_to_monom(monom_m, literal_id, )
            print((source_monom_str, [monom_tuple2str(m) for m in new_monoms]))
            for new_m in new_monoms:
                target_monom_str = monom_tuple2tex_str(new_m)
                graph.add_edge(source_monom_str, target_monom_str,shape = 'circle')
    plt.figure(figsize=(8, 8))
    nx.draw(graph, with_labels=True, font_size=32)
    plt.savefig(save_graph_path, )
    plt.show()


if __name__ == '__main__':
    main()
