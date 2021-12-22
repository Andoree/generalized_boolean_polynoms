import codecs
import os.path
import re
from queue import PriorityQueue
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from generalized_boolean_polynoms.create_polynoms_graph import Polynom
from generalized_boolean_polynoms.utils import TRANSFORMATIONS_VERBOSE, TRANSFORMATIONS_VERBOSE_MASKS, LITERALS, \
    monom_mask_to_str, monom_mask_to_tex_str, TRANSFORMATIONS_VERBOSE_TEX_MASKS, polynom_str_to_tex, split_polynom_str, \
    get_polynom_length_from_str, polynom_str_to_monoms_list, polynom_cyclic_shift, save_paths_dict


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


def calculate_path_extension_statistics(reversed_paths: List, node_id_to_poly_length) -> Dict[int, int]:
    stats_dict = {}
    for path in reversed_paths:
        counter = 0
        for i in range(len(path) - 1):
            node_id_1 = path[i]
            node_id_2 = path[i + 1]
            node_1_poly_length = node_id_to_poly_length[node_id_1]
            node_2_poly_length = node_id_to_poly_length[node_id_2]
            if node_2_poly_length > node_1_poly_length:
                counter += 1
        if stats_dict.get(counter) is None:
            stats_dict[counter] = 0
        stats_dict[counter] += 1

    return stats_dict


def save_stats_dict(stats_dict: Dict, path: str):
    with codecs.open(path, 'w+', encoding="utf-8") as out_file:
        for key, val in stats_dict.items():
            out_file.write(f"{key}\t{val}\n")


def monom_str_to_tuple(monom_str: str) -> Tuple[int]:
    if monom_str.strip() == '-1':
        return -1
    res = tuple((int(x.strip()) for x in monom_str.strip('()').split(',') if x.strip() != ''))
    return res


def filter_paths_dict(paths_to_filter: Dict[int, List[int]], node_index_df):
    polynom_strs = dict()
    for path_id, path in paths_to_filter.items():
        start_node_id = path[0]
        polynom_monoms_list = node_index_df[node_index_df["node_id"] == start_node_id]["polynom_monoms"].values[0]
        num_literals = len(polynom_monoms_list[0])
        met_before = False
        for i in range(1, num_literals):
            cycled_polynom_monoms_list = polynom_cyclic_shift(polynom_monoms_list, n=i)
            poly = Polynom(monoms=cycled_polynom_monoms_list)
            cycled_poly_str = str(poly)
            if polynom_strs.get(cycled_poly_str) is not None:
                met_before = True
                break
        if met_before == False:
            poly = Polynom(polynom_monoms_list)
            poly_str = str(poly)
            polynom_strs[poly_str] = path_id
    keep_path_ids_set = set(polynom_strs.values())
    filtered_paths = [(path_id, path) for path_id, path in paths_to_filter.items() if path_id in keep_path_ids_set]
    return filtered_paths


def create_polynom_text_and_tex_strings(node_index, vertex_id, reversed_path, edges_df):
    s = f"Polynom {node_index[vertex_id]}, length = {len(reversed_path)}: "
    s_tex = f"Polynom ${polynom_str_to_tex(node_index[vertex_id])}$, Number of transformations = {len(reversed_path) - 1}\n\\begin{'{' + 'dmath' + '}'}\n"

    for i in range(len(reversed_path) - 1):
        polynom_source_id = reversed_path[i]
        polynom_dest_id = reversed_path[i + 1]
        transformation_entry = edges_df[
            (edges_df["poly_1_id"] == polynom_source_id) & (edges_df["poly_2_id"] == polynom_dest_id)]

        transform_type_id = transformation_entry["transform_type_id"].values[0]
        transform_edge_literal_id = transformation_entry["literal_id"].values[0]
        transform_edge_monom = transformation_entry["target_monom_mask"].values[0]

        transform_verbose_mask = TRANSFORMATIONS_VERBOSE_MASKS[transform_type_id]
        transform_verbose_tex_mask = TRANSFORMATIONS_VERBOSE_TEX_MASKS[transform_type_id]
        transform_edge_literal = LITERALS[transform_edge_literal_id]
        transform_verbose = transform_verbose_mask.replace("<literal>", transform_edge_literal)
        transform_verbose_tex = transform_verbose_tex_mask.replace("<literal>", transform_edge_literal)
        transform_edge_monom_str = monom_mask_to_str(transform_edge_monom)
        transform_edge_monom_tex_str = monom_mask_to_tex_str(transform_edge_monom)

        polynom_source_verbose = node_index[polynom_source_id]
        polynom_source_verbose_tex = polynom_str_to_tex(polynom_source_verbose)

        s += f"[{polynom_source_verbose}] = [apply {transform_verbose} to {transform_edge_monom_str}] = "
        if len(polynom_source_verbose_tex) > 60:
            first_half_tex, second_half_tex = split_polynom_str(polynom_source_verbose_tex)
            s_tex += fr"{'{'}{first_half_tex} + {'}'} + "
            s_tex += fr"{'{'}{second_half_tex} = [Apply\,({transform_verbose_tex})\,\,to\,\,{transform_edge_monom_tex_str}]{'}'} = "
        else:
            s_tex += fr"{'{'}{polynom_source_verbose_tex} = [Apply\,({transform_verbose_tex})\,\,to\,\,{transform_edge_monom_tex_str}]{'}'} = "
    if len(reversed_path) > 1:
        polynom_dest_verbose = node_index[polynom_dest_id]
        polynom_dest_verbose_tex = polynom_str_to_tex(polynom_dest_verbose)
        s += polynom_dest_verbose
        s_tex += polynom_dest_verbose_tex
    s_tex += "\n\\end{dmath}"
    return s, s_tex


def get_paths_with_expanding_transformations(reversed_paths: List[List[int]], node_id_to_poly_length) -> Dict[
    int, List[int]]:
    expanding_paths = {}
    for (path_source_node_id, reversed_path) in enumerate(reversed_paths):
        for i in range(len(reversed_path) - 1):
            node_id_1 = reversed_path[i]
            node_id_2 = reversed_path[i + 1]
            node_1_poly_length = node_id_to_poly_length[node_id_1]
            node_2_poly_length = node_id_to_poly_length[node_id_2]
            if node_2_poly_length >= node_1_poly_length:
                expanding_paths[path_source_node_id] = reversed_path
    return expanding_paths


def save_longest_reversed_paths(reversed_paths: List[List[int]], edges_df, node_index, save_path: str,
                                save_path_tex: str):
    # print(reversed_paths)
    max_length = max((len(path) for path in reversed_paths))
    longest_paths = {}
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file, \
            codecs.open(save_path_tex, 'w+', encoding="utf-8") as out_file_tex:
        for (vertex_id, reversed_path) in enumerate(reversed_paths):
            if len(reversed_path) == max_length:
                longest_paths[vertex_id] = reversed_path
                s, s_tex = create_polynom_text_and_tex_strings(node_index, vertex_id, reversed_path, edges_df)
                out_file.write(f"{s}\n")
                out_file_tex.write(f"{s_tex}\n")
            # longest_paths.append(longest_paths)
    return longest_paths


def save_longest_reversed_paths_v2(reversed_paths: List[List[int]], edges_df, node_index, save_path: str,
                                   save_path_tex: str):
    max_length = max((len(path) for (idx, path) in reversed_paths))
    longest_paths = {}
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file, \
            codecs.open(save_path_tex, 'w+', encoding="utf-8") as out_file_tex:
        for (vertex_id, reversed_path) in reversed_paths:
            if len(reversed_path) == max_length:
                longest_paths[vertex_id] = reversed_path

                s, s_tex = create_polynom_text_and_tex_strings(node_index, vertex_id, reversed_path, edges_df)
                out_file.write(f"{s}\n")
                out_file_tex.write(f"{s_tex}\n")
    return longest_paths


def save_transformation_paths_strings_and_tex(reversed_paths_dict: List[Tuple[int, List[int]]], edges_df, node_index,
                                              save_path: str,
                                              save_path_tex: str):
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file, \
            codecs.open(save_path_tex, 'w+', encoding="utf-8") as out_file_tex:
        for (vertex_id, reversed_path) in reversed_paths_dict:
            s, s_tex = create_polynom_text_and_tex_strings(node_index, vertex_id, reversed_path, edges_df)
            out_file.write(f"{s}\n")
            out_file_tex.write(f"{s_tex}\n")


def main():
    num_literals = 3
    node_index_path = f"../results/n_{num_literals}/node_index.tsv"
    edges_path = f"../results/n_{num_literals}/edges.tsv"
    output_length_stats_path = f"../results/n_{num_literals}/shortest_paths_length.tsv"
    output_longest_path = f"../results/n_{num_literals}/longest_paths.txt"
    output_longest_path_tex = f"../results/n_{num_literals}/longest_paths.tex"
    output_filtered_longest_path = f"../results/n_{num_literals}/filtered_longest_paths.txt"
    output_filtered_longest_path_tex = f"../results/n_{num_literals}/filtered_longest_paths.tex"
    output_extension_stats_path = f"../results/n_{num_literals}/extensions_stats_length.tsv"
    output_filtered_expanding_path = f"../results/n_{num_literals}/expanding_paths/filtered_expanding_paths.txt"
    output_filtered_expanding_path_tex = f"../results/n_{num_literals}/expanding_paths/filtered_expanding_paths.tex"
    output_filtered_expanding_paths_path = f"../results/n_{num_literals}/expanding_paths/filt_exp_paths_nodex.tsv"
    output_paths = (
        output_length_stats_path, output_longest_path, output_longest_path_tex, output_extension_stats_path,
        output_filtered_longest_path, output_filtered_longest_path_tex, output_filtered_expanding_path,
        output_filtered_expanding_path_tex, output_filtered_expanding_paths_path
    )
    for out_path in output_paths:
        output_dir = os.path.dirname(out_path)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)

    node_index_df = pd.read_csv(node_index_path, sep='\t', header=None, dtype={"node_id": int, "node_verbose": str},
                                names=["node_id", "node_verbose", "polynom_monoms_str"])
    node_index_df["node_poly_length"] = node_index_df["node_verbose"].apply(get_polynom_length_from_str)
    node_index_df["polynom_monoms"] = node_index_df["polynom_monoms_str"].apply(polynom_str_to_monoms_list)

    node_index = node_index_df.set_index("node_id", )["node_verbose"]
    node_id_to_poly_length = node_index_df.set_index("node_id", )["node_poly_length"]
    print("Индекс вершин загружен")
    edges_df = pd.read_csv(edges_path, sep='\t', header=None,
                           names=["poly_1_id", "poly_2_id", "transform_type_id", "literal_id", "target_monom_mask"])
    edges_df["target_monom_mask"] = edges_df["target_monom_mask"].apply(monom_str_to_tuple)
    print("Рёбра загружены")
    adjacency_lists = load_transformations_graph(node_index_df=node_index_df, edges_df=edges_df)
    print("Списки смежности созданы")
    start_vertex_id = 0
    # run_dijkstra(adjacency_lists, 0)
    path_costs, previous_vertices = dijkstra_shortest_path(adjacency_lists, start_vertex_id)
    restored_reversed_paths = restore_reversed_paths(previous_vertices,
                                                     start_vertex_id=start_vertex_id)
    # restored_reversed_paths = [(idx, path) for idx, path in enumerate(restored_reversed_paths)]
    path_length_stats_dict = calculate_path_length_statistics(reversed_paths=restored_reversed_paths)
    save_stats_dict(stats_dict=path_length_stats_dict, path=output_length_stats_path)
    # print_paths(reversed_paths=restored_reversed_paths, edges_df=edges_df, node_index=node_index)
    print("Ищем длиннейшие пути")
    longest_paths = save_longest_reversed_paths(reversed_paths=restored_reversed_paths, edges_df=edges_df,
                                                node_index=node_index,
                                                save_path=output_longest_path, save_path_tex=output_longest_path_tex)

    # print("longest_paths", longest_paths[0][0])
    print("Длиннейшие пути найдены")
    extension_transforms_stats_dict = calculate_path_extension_statistics(reversed_paths=restored_reversed_paths,
                                                                          node_id_to_poly_length=node_id_to_poly_length)
    print("Статистика расширяющих путей посчитана")
    save_stats_dict(stats_dict=extension_transforms_stats_dict, path=output_extension_stats_path)
    print("Статистика расширяющих путей сохранена")
    print(f"Фильтруем длиннейшие пути, их вот столько: {len(longest_paths)}")
    filtered_longest_paths = filter_paths_dict(paths_to_filter=longest_paths, node_index_df=node_index_df)
    print(f"Сохраняем отфильтрованные длиннейшие пути, их вот столько: {len(filtered_longest_paths)}")
    # filtered_longest_paths = list(filtered_longest_paths.values())
    save_longest_reversed_paths_v2(reversed_paths=filtered_longest_paths, edges_df=edges_df,
                                   node_index=node_index,
                                   save_path=output_filtered_longest_path,
                                   save_path_tex=output_filtered_longest_path_tex)
    print("Ищем пути с расширяющими преобразованиями")
    expanding_paths_dict = get_paths_with_expanding_transformations(restored_reversed_paths, node_id_to_poly_length)
    filtered_expanding_paths = filter_paths_dict(paths_to_filter=expanding_paths_dict, node_index_df=node_index_df)
    save_transformation_paths_strings_and_tex(reversed_paths_dict=filtered_expanding_paths,
                                              edges_df=edges_df, node_index=node_index,
                                              save_path=output_filtered_expanding_path,
                                              save_path_tex=output_filtered_expanding_path_tex)
    save_paths_dict(source_id_path_tuples=filtered_expanding_paths, save_path=output_filtered_expanding_paths_path)
    print(f"Сохранили пути с расширяющими преобразованиями, их вот столько: {len(filtered_expanding_paths)}")


if __name__ == '__main__':
    main()
