import codecs
import os.path
from queue import PriorityQueue
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from generalized_boolean_polynoms.utils import TRANSFORMATIONS_VERBOSE


def dijkstra_shortest_path(adjacency_lists: List[List[int]], start_vertex: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(adjacency_lists) == 0:
        raise ValueError("Граф пустой")

    num_vertices = len(adjacency_lists)
    path_costs = np.full(shape=num_vertices, fill_value=np.inf, dtype=np.float)
    previous_vertices = np.full(num_vertices, dtype=np.int, fill_value=-1)
    vertices_to_visit_pq = PriorityQueue()
    # vertices_to_visit = np.full(num_vertices, dtype=np.float, fill_value=np.inf)
    visited_nodes_set = set()
    current_vertex_id = start_vertex
    previous_vertices[current_vertex_id] = current_vertex_id
    vertices_to_visit_pq.put((0, start_vertex))
    # vertices_to_visit[start_vertex] = 0
    path_costs[start_vertex] = 0
    print("Старт алгоритма Дейкстры")
    while not vertices_to_visit_pq.empty():
        if len(visited_nodes_set) % 100000 == 0:
            print(len(visited_nodes_set))
        current_vertex_dist, current_vertex_id = vertices_to_visit_pq.get()
        # current_vertex_id = vertices_to_visit.argmin()

        # TODO: Переписываю
        for neighbor_id in adjacency_lists[current_vertex_id]:
            if neighbor_id not in visited_nodes_set:
                path_cost_to_neighbor_from_current = current_vertex_dist + 1
                if path_cost_to_neighbor_from_current < path_costs[neighbor_id]:
                    previous_vertices[neighbor_id] = current_vertex_id
                    path_costs[neighbor_id] = path_cost_to_neighbor_from_current
                    vertices_to_visit_pq.put((path_cost_to_neighbor_from_current, neighbor_id))

        # for vertex_id, dist in enumerate(weighted_adjacency_matrix[current_vertex_id]):
        #     if vertex_id not in visited_nodes_set and not np.isnan(dist):
        #         path_cost_from_this_node = path_costs[current_vertex_id] + dist
        #         if path_cost_from_this_node < path_costs[vertex_id]:
        #             previous_vertices[vertex_id] = current_vertex_id
        #             path_costs[vertex_id] = path_cost_from_this_node
        #             vertices_to_visit[vertex_id] = path_cost_from_this_node
        visited_nodes_set.add(current_vertex_id)
        # vertices_to_visit[current_vertex_id] = np.inf
    if len(visited_nodes_set) < num_vertices:
        raise ValueError("Граф не является связным")

    return path_costs, previous_vertices


def restore_reversed_paths(previous_vertices: np.ndarray, start_vertex_id: int) -> List[List[int]]:
    """
    Метод восстанавливает полные минимальные пути до целевых вершин, но пути инвертированные.
    :param previous_vertices: Массив, размерность которого равна числу вершин. Каждый элемент массива - это номер
    предпоследней вершины на минимальном пути в заданную.
    :param start_vertex_id: Стартовая вершина - та, из которой ищутся минимальные пути.
    :return: Список инвертированных минимальных путей в целевые вершины. Каждый путь - это последовательность вершин,
    через которые проходит минимальный путь, то есть все вершины, которые нужно обойти на минимальном пути.
    """
    reversed_paths = []
    for destination_vertex_id in range(len(previous_vertices)):
        path = []
        current_vertex = destination_vertex_id
        path.append(current_vertex)
        while current_vertex != start_vertex_id:
            current_vertex = previous_vertices[current_vertex]
            path.append(current_vertex)
        reversed_paths.append(path)

    return reversed_paths


def print_paths(reversed_paths: List[List[int]], edges_df, node_index):
    """
    Метод получает на вход список инвертированных путей до целевых вершин, инвертирует их повторно,
    чтобы получить прямые пути и выводит эти пути. Путь - это последовательность номеров вершин, через
    которые проходит путь
    :param reversed_paths: Список инвертированных путей
    """
    for vertex_id, reversed_path in enumerate(reversed_paths):
        s = f"путь длины: {len(reversed_path)}:"
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


def run_dijkstra(adjacency_lists: List[List[int]], start_vertex_id: int):
    """
    Функция, запускающая алгоритм Дейкстры для заданных матрицы и стартовой вершины
    :param adjacency_lists: Списки смежности графа
    :param start_vertex_id: Стартовая вершина
    """
    try:
        path_costs, previous_vertices = dijkstra_shortest_path(adjacency_lists, start_vertex_id)
        print(f"Стартовая вершина: {start_vertex_id}. Расстояния до вершин от неё:")
        print(path_costs)
        # print("Предыдущие вершины на пути в заданные:")
        # print(previous_vertices)
        restored_reversed_paths = restore_reversed_paths(previous_vertices,
                                                         start_vertex_id=start_vertex_id)
        print("Выведем минимальные пути:")
        print_paths(restored_reversed_paths)
    except ValueError as e:
        print(f"Ошибка, текст ошибки:\n{str(e)}")
    print('-' * 10)


def load_transformations_graph(node_index_df, edges_df):
    num_vertices = node_index_df.shape[0]
    adjacency_lists = [[] for _ in range(num_vertices)]
    # TODO: Надо поменять начало и конец ребра местами, тогда обход из
    # TODO: нуля по обратным рёбрам - именно то, что нужно
    # print("aaa", int(edges_df.shape[0] / 100))
    for idx, row in tqdm(edges_df.iterrows(), total=edges_df.shape[0], miniters=int(edges_df.shape[0] / 100)):
        poly_source_id = row["poly_1_id"]
        poly_target_id = row["poly_2_id"]
        # print("source", poly_source_id)
        # print("target",poly_target_id)
        adjacency_lists[poly_target_id].append(poly_source_id)
    return adjacency_lists


def calculate_path_length_statistics(reversed_paths: List) -> Dict[int, int]:
    stats_dict = {}
    for path in reversed_paths:
        path_length = len(path)
        if stats_dict.get(path_length) is None:
            stats_dict[path_length] = 0
        stats_dict[path_length] += 1
    return stats_dict


def save_stats_dict(stats_dict: Dict, path: str):
    with codecs.open(path, 'w+', encoding="utf-8") as out_file:
        for key, val in stats_dict.items():
            out_file.write(f"{key}\t{val}\n")


def main():
    node_index_path = "../results/n_3/node_index.tsv"
    edges_path = "../results/n_3/edges.tsv"
    output_length_stats_path = "../results/n_3/shortest_paths_length.tsv"
    output_dir = os.path.dirname(output_length_stats_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    node_index_df = pd.read_csv(node_index_path, sep='\t', header=None, dtype={"node_id": int, "node_verbose": str},
                                names=["node_id", "node_verbose"])
    node_index = node_index_df.set_index("node_id", )["node_verbose"]
    print("Индекс вершин загружен")
    edges_df = pd.read_csv(edges_path, sep='\t', header=None, names=["poly_1_id", "poly_2_id", "transform_type_id"])
    print("Рёбра загружены")
    adjacency_lists = load_transformations_graph(node_index_df=node_index_df, edges_df=edges_df)
    print("Списки смежности созданы")
    start_vertex_id = 0
    # run_dijkstra(adjacency_lists, 0)
    path_costs, previous_vertices = dijkstra_shortest_path(adjacency_lists, start_vertex_id)
    restored_reversed_paths = restore_reversed_paths(previous_vertices,
                                                     start_vertex_id=start_vertex_id)
    stats_dict = calculate_path_length_statistics(reversed_paths=restored_reversed_paths)
    save_stats_dict(stats_dict=stats_dict, path=output_length_stats_path)
    # print_paths(reversed_paths=restored_reversed_paths, edges_df=edges_df, node_index=node_index)


if __name__ == '__main__':
    main()
