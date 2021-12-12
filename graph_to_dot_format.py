import codecs
import os.path

import pandas as pd

from generalized_boolean_polynoms.utils import TRANSFORMATIONS_VERBOSE


def graph_to_dot(node_id_to_verbose: pd.Series, edges: pd.DataFrame):
    dot_representation = 'graph G {\n'
    for idx, row in edges.iterrows():
        poly_1_id = row["poly_1_id"]
        poly_2_id = row["poly_2_id"]
        transform_type_id = row["transform_type_id"]
        poly_1_verbose = node_id_to_verbose[poly_1_id]
        poly_2_verbose = node_id_to_verbose[poly_2_id]
        transform_type_verbose = TRANSFORMATIONS_VERBOSE[transform_type_id]
        edge_dot_representation = f"\"{poly_1_verbose}\" -- \"{poly_2_verbose}\" [ label=\"{transform_type_verbose}\" ];\n"
        dot_representation += edge_dot_representation
    dot_representation += "}"
    return dot_representation
        # print(poly_1_verbose, '|||', poly_2_verbose, '|||', transform_type_verbose)




def main():
    node_index_path = "../results/n_1/node_index.tsv"
    edges_path = "../results/n_1/edges.tsv"
    output_dot_path = "../results/n_1/graph.dot"
    output_dir = os.path.dirname(output_dot_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    node_index_df = pd.read_csv(node_index_path, sep='\t', header=None, dtype={"node_id": int, "node_verbose": str},
    names = ["node_id", "node_verbose"])
    edges_df = pd.read_csv(edges_path, sep='\t', header=None, names=["poly_1_id", "poly_2_id", "transform_type_id"])
    print(node_index_df.shape)
    # print('--')
    # print(edges_df)
    print(node_index_df.set_index("node_id", ))
    print(node_index_df.set_index("node_id", ).dtypes)
    node_index = node_index_df.set_index("node_id", )["node_verbose"]
    print(node_index)
    print(type(node_index))
    # print(node_index[31])
    # print('aaa', node_index[0])
    # print(node_index.to_dict(orient=1))


    graph_dot_representation = graph_to_dot(node_id_to_verbose=node_index, edges=edges_df)
    with codecs.open(output_dot_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"{graph_dot_representation}\n")


if __name__ == '__main__':
    main()
