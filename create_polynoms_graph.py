import codecs
import os.path
import time
from queue import PriorityQueue
from typing import Tuple, List, Dict
from generalized_boolean_polynoms.utils import LITERALS, polynom_monoms_list_to_str
from generalized_boolean_polynoms.classes import Polynom, Transformation
from generalized_boolean_polynoms.transform import apply_transformation_to_monom

"""
    Какие компоненты мне здесь нужны?
    1. Мне нужно как-то задать полином. Пока что думаю, что это будет список мономов
    2. Следовательно, у меня должен быть моном. Что такое моном?
    Моном - это произведение литералов. При этом моном от n переменных - это набор длины
    n, состоящий из -1, 0, 1
    3. Мне надо организовать перебор преобразований следующим образом:
        3.1 Пройтись по всем мономам полинома. К каждому попробовать применить одно
        из трёх преобразований
        3.2 # TODO: как понять, какой полином получится после преобразования?
        Ну по идее я наверное могу описать, как влияет каждое из преобразований на моном.
        В преобразовании я трогаю лишь одну переменную, но она могла домножиться на другую.
        Тут достаточно сложно, надо ещё думать.
        3.3 Возможно, мне надо научиться раскрывать скобки и приводить подобные слагаемые.
        #TODO: Подумай, как каждое из преобразований влияет на появление мономов:
        1) x = -x + 1 - добавляет моном, где вместо x стоит -x и моном без x
        2) 0 = x +(-x) + 1 - добавляет 3 монома, да? TODO
        3) 1 = x + -x - добавляет 2 монома, в которых теперь есть переменная, которой не было. 
    4. Мне нужен класс "Преобразование" - это отношение между вершинами графа(полиномами)
"""


def process_new_polynom(new_polynom_monoms: List[Tuple[int]], new_monoms: List[Tuple[int]],
                        existing_polynom_nodes: Dict[str, Polynom], traverse_pq: PriorityQueue,
                        current_polynom: Polynom, transform_type: int, literal_id: int,
                        processed_monom_mask, current_dist: int):
    # Добавляем новые мономы в новый полином
    new_polynom_monoms.extend(new_monoms)
    # Создаём новый полином по результатам преобразования
    new_polynom = Polynom(new_polynom_monoms)
    # Проверяем, что полинома, полученного преобразованием, ещё не обходили
    if existing_polynom_nodes.get(str(new_polynom)) is None:
        # Если не обходили, то запоминаем, что теперь планируем обход
        existing_polynom_nodes[str(new_polynom)] = new_polynom
        # Если не обходили, включаем в очередь на обход.
        traverse_pq.put((current_dist + 1, time.time(), new_polynom))
        # traverse_queue.append(new_polynom)
    else:
        new_polynom = existing_polynom_nodes[str(new_polynom)]
    transformation_edge = Transformation(source_poly=current_polynom, dest_poly=new_polynom,
                                         transform_type=transform_type, literal_id=literal_id,
                                         processed_monom_mask=processed_monom_mask)
    return transformation_edge


def update_transformation_edges_dict(polynom_transformation_edges: Dict[str, Dict[str, Transformation]],
                                     transformation_edge: Transformation):
    edge_source = transformation_edge.source_poly
    edge_dest = transformation_edge.dest_poly
    transform_type = transformation_edge.transform_type
    if polynom_transformation_edges.get(str(edge_source)) is None:
        polynom_transformation_edges[str(edge_source)] = {}
    if polynom_transformation_edges[str(edge_source)].get(str(edge_dest)) is None:
        polynom_transformation_edges[str(edge_source)][str(edge_dest)] = transformation_edge


def transformations_brute_force(num_literals: int, initial_polynom: Polynom):
    # traverse_queue = [initial_polynom]
    traverse_queue_pq = PriorityQueue()
    traverse_queue_pq.put((0, time.time(), initial_polynom))
    existing_polynom_nodes = {str(initial_polynom): initial_polynom}
    min_path_length_stats_dict = {}
    polynom_transformation_edges = {}
    # Маппинг из строки в объект класса полином.
    literals = LITERALS[:num_literals]
    while not traverse_queue_pq.empty():
        if len(existing_polynom_nodes) % 1000 == 0:
            print(len(existing_polynom_nodes))
        # current_polynom = traverse_queue.pop(0)
        current_vertex_dist, _, current_polynom = traverse_queue_pq.get()
        if min_path_length_stats_dict.get(current_vertex_dist) is None:
            min_path_length_stats_dict[current_vertex_dist] = 0
        min_path_length_stats_dict[current_vertex_dist] += 1
        polynom_monoms = current_polynom.monoms
        for monom_mask in polynom_monoms:
            num_literals = len(monom_mask)
            for literal_id in range(num_literals):
                # Заводим копию полинома
                new_polynom_monoms = list(current_polynom.monoms)
                new_polynom_monoms.remove(monom_mask)
                new_monoms, transform_type = apply_transformation_to_monom(monom_mask, literal_id, )
                transformation_edge = process_new_polynom(new_polynom_monoms=new_polynom_monoms, new_monoms=new_monoms,
                                                          existing_polynom_nodes=existing_polynom_nodes,
                                                          traverse_pq=traverse_queue_pq,
                                                          current_polynom=current_polynom,
                                                          transform_type=transform_type,
                                                          current_dist=current_vertex_dist,
                                                          literal_id=literal_id, processed_monom_mask=monom_mask)
                update_transformation_edges_dict(polynom_transformation_edges=polynom_transformation_edges,
                                                 transformation_edge=transformation_edge, )

        # Добавляем представление нуля
        for literal_id in range(num_literals):
            # Заводим копию полинома
            new_polynom_monoms = list(current_polynom.monoms)
            new_monoms, transform_type = apply_transformation_to_monom(monom=None, literal_id=literal_id,
                                                                       num_literals=num_literals)
            transformation_edge = process_new_polynom(new_polynom_monoms=new_polynom_monoms, new_monoms=new_monoms,
                                                      existing_polynom_nodes=existing_polynom_nodes,
                                                      traverse_pq=traverse_queue_pq,
                                                      current_polynom=current_polynom,
                                                      transform_type=transform_type, current_dist=current_vertex_dist,
                                                      literal_id=literal_id, processed_monom_mask=-1)
            update_transformation_edges_dict(polynom_transformation_edges=polynom_transformation_edges,
                                             transformation_edge=transformation_edge, )
    return existing_polynom_nodes, polynom_transformation_edges, min_path_length_stats_dict


def save_graph(polynom_nodes: Dict[str, Polynom], polynom_transformation_edges: Dict[str, Dict[str, Transformation]],
               node_index_path: str, edges_path: str):
    polynom_verbose_to_id = {poly_str: idx for idx, poly_str in enumerate(polynom_nodes.keys())}
    with codecs.open(node_index_path, 'w+', encoding="utf-8") as node_index_file:
        for poly_str, idx in polynom_verbose_to_id.items():
            polynom_object = polynom_nodes[poly_str]
            polynom_monom_masks_str = polynom_monoms_list_to_str(monoms_list=polynom_object.monoms)
            node_index_file.write(f"{idx}\t{str(poly_str)}\t{polynom_monom_masks_str}\n")
    with codecs.open(edges_path, 'w+', encoding="utf-8") as edges_file:
        for poly_1_node_str, poly_1_dict in polynom_transformation_edges.items():
            for poly_2_node_str, transformation_edge in poly_1_dict.items():
                transform_type = transformation_edge.transform_type
                literal_id = transformation_edge.literal_id
                monom_mask = transformation_edge.processed_monom_mask
                edges_file.write(f"{polynom_verbose_to_id[poly_1_node_str]}\t"
                                 f"{polynom_verbose_to_id[poly_2_node_str]}\t"
                                 f"{transform_type}\t"
                                 f"{literal_id}\t"
                                 f"{monom_mask}\n")


def main():
    num_literals = 3
    node_index_path = f"results/n_{num_literals}/node_index.tsv"
    edges_path = f"results/n_{num_literals}/edges.tsv"
    output_dir = os.path.dirname(node_index_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_dir = os.path.dirname(edges_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    initial_polynom = Polynom(list())
    print("INITIAL", initial_polynom)
    polynom_nodes, polynom_transformation_edges, min_path_length_stats_dict = transformations_brute_force(
        num_literals=num_literals,
        initial_polynom=initial_polynom)
    save_graph(polynom_nodes=polynom_nodes, polynom_transformation_edges=polynom_transformation_edges,
               node_index_path=node_index_path, edges_path=edges_path)
    print("Статистика длин путей:")
    for k, v in min_path_length_stats_dict.items():
        print(f"{k}: {v}")
    # print(len(polynom_nodes))
    # print("Число вершин", polynom_nodes.keys())
    # print('--')
    # # print("Число рёбер", len(polynom_transformation_edges))
    # print(polynom_transformation_edges)
    # i = 0
    # for poly_1_node, poly_1_dict in polynom_transformation_edges.items():
    #     for poly_2_node, transform_type in poly_1_dict.items():
    #         i += 1
    #         print(f"[{poly_1_node}] --- {TRANSFORMATIONS_VERBOSE[transform_type]} --- [{poly_2_node}]")
    # print("Число рёбер", i)


if __name__ == '__main__':
    main()
