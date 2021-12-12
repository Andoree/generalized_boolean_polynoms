import codecs
import os.path
from collections import Counter
from typing import Tuple, List, Dict
import numpy as np

LITERALS = ['x', 'y', 'z']
TRANSFORMATIONS_VERBOSE = {
    1: "x = -x + 1",
    2: "0 = x +(-x) + 1",
    0: "1 = x + -x",
    -1: "-x = x + 1"
}

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


class Polynom:
    def __init__(self, monoms, ):
        # print(monoms)
        monoms = [tuple(t) for t in monoms]
        # print(monoms)
        # print('-')
        monom_counter = Counter(monoms)
        filtered_monoms = [monom for monom, count in monom_counter.items() if count % 2 == 1]
        # print("filtered", filtered_monoms)
        # print('-')
        filtered_monoms.sort()
        self.monoms = filtered_monoms

    def __str__(self):
        monoms_strs = []
        if len(self.monoms) > 0:
            for monom in self.monoms:
                num_literals = len(monom)
                s = ''
                for lit, mask_value in zip(LITERALS[:num_literals], monom):
                    if mask_value == 1:
                        s += lit
                    elif mask_value == -1:
                        s += f'(-{lit})'
                if s == '':
                    s = '1'
                monoms_strs.append(s)
            return " + ".join(monoms_strs)
        else:
            return "0"

    def __repr__(self):
        return self.__str__()


class Transformation:

    def __init__(self, source_poly: Polynom, dest_poly: Polynom, transform_type):
        self.source_poly = source_poly
        self.dest_poly = dest_poly
        self.transform_type = transform_type

    def __str__(self):
        return f"[{self.source_poly}] --- {TRANSFORMATIONS_VERBOSE[self.transform_type]} ---> [{self.dest_poly}]"

    def __repr__(self):
        return self.__str__()


def apply_transformation_to_monom(monom, literal_id, num_literals=None):
    if monom is None:
        assert num_literals is not None
        plus_x_monom, minus_x_monom, one_x_monom = [0, ] * num_literals, \
                                                   [0, ] * num_literals, \
                                                   [0, ] * num_literals
        plus_x_monom[literal_id] = 1
        minus_x_monom[literal_id] = -1
        one_x_monom[literal_id] = 0
        monoms = [plus_x_monom, minus_x_monom, one_x_monom]
        transform_type = 2
    else:
        literal_mask_val = monom[literal_id]
        if literal_mask_val == -1:
            # Случай, когда переменная в мономе присутствует с отрицанием
            plus_x_monom, one_x_monom = list(monom), list(monom)  # .copy()
            plus_x_monom[literal_id] = 1
            one_x_monom[literal_id] = 0
            monoms = [plus_x_monom, one_x_monom]
            transform_type = -1
        elif literal_mask_val == 0:
            # Случай, когда переменная в мономе не присутствует (не значима, равна 1)
            plus_x_monom, minus_x_monom = list(monom), list(monom)  # monom.copy(), monom.copy()
            plus_x_monom[literal_id] = 1  # min(x_monom[literal_id] + 1, 1)
            minus_x_monom[literal_id] = -1
            monoms = [plus_x_monom, minus_x_monom]
            transform_type = 0
        elif literal_mask_val == 1:
            # Случай, когда переменная в мономе присутствует без отрицания
            minus_x_monom, one_x_monom = list(monom), list(monom)  # monom.copy(), monom.copy()
            minus_x_monom[literal_id] = -1
            one_x_monom[literal_id] = 0
            monoms = [minus_x_monom, one_x_monom]
            transform_type = 1
        else:
            raise ValueError(f"Возможные значения маски монома: -1, 0, 1. Получено: {literal_mask_val}")

        """
        1: "x = -x + 1",
        0: "1 = x + -x",
        -1: "-x = x + 1"
        """
    return monoms, transform_type


def process_new_polynom(new_polynom_monoms: List[Tuple[int]], new_monoms: List[Tuple[int]],
                        existing_polynom_nodes: Dict[str, Polynom], traverse_queue: List[Polynom],
                        current_polynom: Polynom, transform_type: int):
    # Добавляем новые мономы в новый полином
    new_polynom_monoms.extend(new_monoms)
    # Создаём новый полином по результатам преобразования
    new_polynom = Polynom(new_polynom_monoms)
    # Проверяем, что полинома, полученного преобразованием, ещё не обходили
    if existing_polynom_nodes.get(str(new_polynom)) is None:
        # Если не обходили, то запоминаем, что теперь планируем обход
        existing_polynom_nodes[str(new_polynom)] = new_polynom
        # Если не обходили, включаем в очередь на обход.
        traverse_queue.append(new_polynom)
    else:
        new_polynom = existing_polynom_nodes[str(new_polynom)]
    transformation_edge = Transformation(source_poly=current_polynom, dest_poly=new_polynom,
                                         transform_type=transform_type)
    return transformation_edge


def update_transformation_edges_dict(polynom_transformation_edges: Dict[str, Dict[str, Transformation]],
                                     transformation_edge: Transformation):
    edge_source = transformation_edge.source_poly
    edge_dest = transformation_edge.dest_poly
    transform_type = transformation_edge.transform_type
    if polynom_transformation_edges.get(str(edge_source)) is None:
        polynom_transformation_edges[str(edge_source)] = {}
    if polynom_transformation_edges.get(str(edge_dest)) is None:
        polynom_transformation_edges[str(edge_dest)] = {}
    if polynom_transformation_edges[str(edge_source)].get(str(edge_dest)) is None and \
            polynom_transformation_edges[str(edge_dest)].get(str(edge_source)) is None:
        polynom_transformation_edges[str(edge_source)][str(edge_dest)] = transform_type


def transformations_brute_force(num_literals: int, initial_polynom: Polynom):
    traverse_queue = [initial_polynom]
    existing_polynom_nodes = {str(initial_polynom): initial_polynom}
    polynom_transformation_edges = {}
    # Маппинг из строки в объект класса полином.
    literals = LITERALS[:num_literals]
    while len(traverse_queue) > 0:
        if len(existing_polynom_nodes) % 1000 == 0:
            print(len(existing_polynom_nodes))
        current_polynom = traverse_queue.pop(0)
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
                                                          traverse_queue=traverse_queue,
                                                          current_polynom=current_polynom,
                                                          transform_type=transform_type)
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
                                                      traverse_queue=traverse_queue,
                                                      current_polynom=current_polynom,
                                                      transform_type=transform_type)
            update_transformation_edges_dict(polynom_transformation_edges=polynom_transformation_edges,
                                             transformation_edge=transformation_edge, )
    return existing_polynom_nodes, polynom_transformation_edges


def save_graph(polynom_nodes: Dict[str, Polynom], polynom_transformation_edges: Dict[str, Dict[str, Transformation]],
               node_index_path: str, edges_path: str):
    polynom_verbose_to_id = {poly_str: idx for idx, poly_str in enumerate(polynom_nodes.keys())}
    with codecs.open(node_index_path, 'w+', encoding="utf-8") as node_index_file:
        for poly_str, idx in polynom_verbose_to_id.items():
            node_index_file.write(f"{idx}\t{str(poly_str)}\n")
    with codecs.open(edges_path, 'w+', encoding="utf-8") as edges_file:
        for poly_1_node_str, poly_1_dict in polynom_transformation_edges.items():
            for poly_2_node_str, transform_type in poly_1_dict.items():
                edges_file.write(f"{poly_1_node_str}\t{poly_2_node_str}\t{transform_type}\n")


def main():
    node_index_path = "results/node_index.txt"
    edges_path = "results/edges.txt"
    output_dir = os.path.dirname(node_index_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_dir = os.path.dirname(edges_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    initial_polynom = Polynom(list())
    print("INITIAL", initial_polynom)
    polynom_nodes, polynom_transformation_edges = transformations_brute_force(num_literals=3,
                                                                              initial_polynom=initial_polynom)
    save_graph(polynom_nodes=polynom_nodes, polynom_transformation_edges=polynom_transformation_edges,
               node_index_path=node_index_path, edges_path=edges_path)
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
