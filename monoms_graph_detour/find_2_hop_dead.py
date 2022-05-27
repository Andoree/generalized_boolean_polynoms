import codecs
import itertools
from typing import List, Set

from tqdm import tqdm

from generalized_boolean_polynoms.classes import Polynom
from generalized_boolean_polynoms.monom_graph_analysis.build_monoms_graph import create_monoms_graph_adjacency_lists

# TODO: 2-тупиковость реализовать. Это пока 1-тупиковость
from generalized_boolean_polynoms.utils import monom_tuple2str


def traverse_monoms_graph_two_hop(adjacency_lists):
    num_monoms = len(adjacency_lists)
    dead_polynoms_pool: Set[str] = set()
    for start_node_id in tqdm(range(num_monoms), total=num_monoms):
        start_visited_nodes: Set[int] = set()
        start_excluded_nodes: Set[int] = set()

        start_neighbors = set(adjacency_lists[start_node_id])
        start_visited_nodes.add(start_node_id)
        # Исключаем из кандидатов на посещение соседей стартовой вершины и саму стартовую
        start_excluded_nodes = start_excluded_nodes.union(start_neighbors)
        start_excluded_nodes.add(start_node_id)
        for neighbor_id in start_neighbors:
            neighbors_of_neighbor = set(adjacency_lists[neighbor_id])
            start_excluded_nodes = start_excluded_nodes.union(neighbors_of_neighbor)

        visit_polynom_node_two_hop(adjacency_lists=adjacency_lists, visited_nodes=start_visited_nodes,
                                   excluded_nodes=start_excluded_nodes, num_nodes=num_monoms,
                                   dead_polynoms_strs_pool=dead_polynoms_pool)
    return dead_polynoms_pool


def traverse_monoms_graph(adjacency_lists):
    num_monoms = len(adjacency_lists)
    dead_polynoms_pool: Set[str] = set()
    for start_node_id in tqdm(range(num_monoms), total=num_monoms):
        start_visited_nodes: Set[int] = set()
        start_excluded_nodes: Set[int] = set()

        start_neighbors = set(adjacency_lists[start_node_id])
        start_visited_nodes.add(start_node_id)
        # Исключаем из кандидатов на посещение соседей стартовой вершины и саму стартовую
        start_excluded_nodes = start_excluded_nodes.union(start_neighbors)
        start_excluded_nodes.add(start_node_id)

        visit_polynom_node(adjacency_lists=adjacency_lists, visited_nodes=start_visited_nodes,
                           excluded_nodes=start_excluded_nodes, num_nodes=num_monoms,
                           dead_polynoms_strs_pool=dead_polynoms_pool)
    return dead_polynoms_pool


def visit_polynom_node(adjacency_lists, visited_nodes: Set[int], excluded_nodes: Set[int], num_nodes: int,
                       dead_polynoms_strs_pool: Set[str]):
    nodes_to_visit = [node_id for node_id in range(num_nodes) if node_id not in excluded_nodes]
    if len(dead_polynoms_strs_pool) % 100000 == 0 and len(dead_polynoms_strs_pool) != 0:
        print(len(dead_polynoms_strs_pool))
    for next_hop_node_id in nodes_to_visit:
        # В список уже посещённых полиномов передаёт посещённые ранее + следующую обрабатываемую
        next_hop_visited_nodes = visited_nodes.copy()
        next_hop_visited_nodes.add(next_hop_node_id)
        sorted_next_hop_visited_nodes = sorted(next_hop_visited_nodes)
        next_hop_visited_nodes_str = "_".join((str(x) for x in sorted_next_hop_visited_nodes))
        if next_hop_visited_nodes_str in dead_polynoms_strs_pool:
            continue

        # Исключаем из кандидатов на посещение соседей обрабатываемой вершины
        next_hop_neighbors = set(adjacency_lists[next_hop_node_id])
        next_hop_excluded_nodes = excluded_nodes.copy()
        next_hop_excluded_nodes = next_hop_excluded_nodes.union(next_hop_neighbors)
        # Исключаем также и обрабатываемую на следующем шаге вершину
        next_hop_excluded_nodes.add(next_hop_node_id)

        # Рекурсивный обход
        visit_polynom_node(adjacency_lists=adjacency_lists, visited_nodes=next_hop_visited_nodes,
                           excluded_nodes=next_hop_excluded_nodes, num_nodes=num_nodes,
                           dead_polynoms_strs_pool=dead_polynoms_strs_pool)
    sorted_visited_nodes = sorted(visited_nodes)
    sorted_visited_nodes_str = "_".join((str(x) for x in sorted_visited_nodes))
    dead_polynoms_strs_pool.add(sorted_visited_nodes_str)


def visit_polynom_node_two_hop(adjacency_lists, visited_nodes: Set[int], excluded_nodes: Set[int], num_nodes: int,
                               dead_polynoms_strs_pool: Set[str]):
    nodes_to_visit = [node_id for node_id in range(num_nodes) if node_id not in excluded_nodes]
    # if len(dead_polynoms_strs_pool) % 100000 == 0 and len(dead_polynoms_strs_pool) != 0:
    #     print(len(dead_polynoms_strs_pool))
    for next_hop_node_id in nodes_to_visit:
        # В список уже посещённых полиномов передаёт посещённые ранее + следующую обрабатываемую
        next_hop_visited_nodes = visited_nodes.copy()
        next_hop_visited_nodes.add(next_hop_node_id)
        sorted_next_hop_visited_nodes = sorted(next_hop_visited_nodes)
        next_hop_visited_nodes_str = "_".join((str(x) for x in sorted_next_hop_visited_nodes))
        if next_hop_visited_nodes_str in dead_polynoms_strs_pool:
            continue

        # Исключаем из кандидатов на посещение соседей обрабатываемой вершины
        next_hop_neighbors = set(adjacency_lists[next_hop_node_id])
        next_hop_excluded_nodes = excluded_nodes.copy()
        next_hop_excluded_nodes = next_hop_excluded_nodes.union(next_hop_neighbors)
        # Исключаем также и обрабатываемую на следующем шаге вершину
        next_hop_excluded_nodes.add(next_hop_node_id)

        for neighbor_id in next_hop_neighbors:
            neighbors_of_neighbor = set(adjacency_lists[neighbor_id])
            next_hop_excluded_nodes = next_hop_excluded_nodes.union(neighbors_of_neighbor)

        # Рекурсивный обход
        visit_polynom_node_two_hop(adjacency_lists=adjacency_lists, visited_nodes=next_hop_visited_nodes,
                                   excluded_nodes=next_hop_excluded_nodes, num_nodes=num_nodes,
                                   dead_polynoms_strs_pool=dead_polynoms_strs_pool)
    sorted_visited_nodes = sorted(visited_nodes)
    sorted_visited_nodes_str = "_".join((str(x) for x in sorted_visited_nodes))
    dead_polynoms_strs_pool.add(sorted_visited_nodes_str)


def count_significant_variables(polynom_monoms_list, num_literals):
    unique_variables_set = set()
    for monom_mask in polynom_monoms_list:
        for literal_id, literal_value in enumerate(monom_mask):
            if literal_value == 1 or literal_value == -1:
                unique_variables_set.add(literal_id)
                if len(unique_variables_set) == num_literals:
                    return len(unique_variables_set)
    return len(unique_variables_set)


def calculate_polynom_value_vector(input_sets, polynom_monoms_list, num_literals):
    function_value_vector = []

    for inp_set in input_sets:
        monoms_sum = 0
        for monom in polynom_monoms_list:
            monom_value = 1
            assert len(monom) == len(inp_set)
            for literal_id in range(num_literals):
                input_x = inp_set[literal_id]
                literal_polarity = monom[literal_id]
                # 0 тогда, когда вход = 0, литерал в мономе положительный(=1)
                # и тогда, когда вход = 1, литерал в мономе положительный(= -1)
                if (input_x == 0 and literal_polarity == 1) or (input_x == 1 and literal_polarity == -1):
                    monom_value = 0
            monoms_sum += monom_value

        func_val = monoms_sum % 2
        function_value_vector.append(func_val)

    return function_value_vector


# TODO: Сет решений - это строки. Строки - упорядоченные номера вершин. Проверять, что
# TODO: такой набор мономов ещё не встречался
def main():
    num_literals = 3
    # save_graph_path = f"../../results/n_{num_literals}/monoms_graph/monom_transformations_graph.png"
    # output_dir = os.path.dirname(save_graph_path)
    # if not os.path.exists(output_dir) and output_dir != '':
    #     os.makedirs(output_dir)
    adjacency_lists, monom2node_id = create_monoms_graph_adjacency_lists(num_literals=num_literals)
    node_id2monom = {node_id: monom for monom, node_id in monom2node_id.items()}



    two_hop_dead_polynom_strs_set = traverse_monoms_graph_two_hop(adjacency_lists)
    print(f"2-тупиковых полиномов от {num_literals} переменных: {len(two_hop_dead_polynom_strs_set)}")
    monom_strs_list = []
    filtered_monom_masks = []
    for dead_poly_str in two_hop_dead_polynom_strs_set:
        monom_ids = [int(x) for x in dead_poly_str.split('_')]
        monom_masks = [node_id2monom[node_id] for node_id in monom_ids]
        sign_variables = count_significant_variables(monom_masks, num_literals=num_literals)
        if sign_variables < num_literals:
            continue
        elif sign_variables > num_literals:
            raise Exception()
        filtered_monom_masks.append(monom_masks)
        monom_strs = [monom_tuple2str(mm) for mm in monom_masks]
        monom_strs_list.append(monom_strs)
    monom_strs_list.sort(key=lambda x: len(x))
    print(f"2-тупиковых полиномов от {num_literals} переменных после фильтрации: {len(monom_strs_list)}")
    for i, monom_strs in enumerate(monom_strs_list):
        print(i, monom_strs)
    input_sets = list(itertools.product((0, 1), repeat=num_literals))
    value_vec2monoms_list = {}
    print('---')
    for filtered_polynom_monoms in filtered_monom_masks:
        value_vector = calculate_polynom_value_vector(input_sets=input_sets, num_literals=num_literals,
                                                      polynom_monoms_list=filtered_polynom_monoms)
        val_vector_str = "".join((str(x) for x in value_vector))
        if value_vec2monoms_list.get(val_vector_str) is None:
            value_vec2monoms_list[val_vector_str] = []
        value_vec2monoms_list[val_vector_str].append(filtered_polynom_monoms)
    i = 0
    for val_vector, polynoms_list in value_vec2monoms_list.items():
        unique_poly_lengths = set((len(mm) for mm in polynoms_list))
        if len(polynoms_list) > 1 and len(unique_poly_lengths) > 1:
            i += 1
            monom_strs_list = [str(Polynom(mm)) for mm in polynoms_list]
            print(i, val_vector, len(polynoms_list), unique_poly_lengths, monom_strs_list,)  # " || ".join(monom_strs_list))

    dead_polynom_strs_set = traverse_monoms_graph(adjacency_lists)

    print(f"Тупиковых полиномов от {num_literals} переменных: {len(dead_polynom_strs_set)}")
    monom_strs_list = []
    i = 0
    for dead_poly_str in dead_polynom_strs_set:
        monom_ids = [int(x) for x in dead_poly_str.split('_')]
        monoms = [node_id2monom[node_id] for node_id in monom_ids]
        sign_variables = count_significant_variables(monoms, num_literals=num_literals)
        if sign_variables != num_literals:
            continue
        i += 1
        monom_strs = [monom_tuple2str(node_id2monom[node_id]) for node_id in monom_ids]
        monom_strs_list.append(monom_strs)
    monom_strs_list.sort(key=lambda x: len(x))
    with codecs.open(f"dead_polynoms_n_{num_literals}.txt", 'w+', encoding="utf-8") as out_file:
        for i, monom_strs in  tqdm(enumerate(monom_strs_list)):
            out_file.write(f"{i + 1}: {' + '.join(monom_strs)}\n")
    # for i, monom_strs in enumerate(monom_strs_list):
    #     print(i + 1, monom_strs)
    print(f"Тупиковых полиномов существенно зависящих от {num_literals} переменных: {len(monom_strs_list)}")
    # 0 переменных - 2 тупиковых (0 и 1)
    # 1 переменной - 2 тупиковых (x и -x)
    # 2 переменных - 34 тупиковых (следовательно, 30)


if __name__ == '__main__':
    main()
