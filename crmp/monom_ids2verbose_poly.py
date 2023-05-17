import codecs

from generalized_boolean_polynoms.practice_sem_3.utils import read_shortest_poly_monom_ids, read_monom_id2str, \
    int2binary_string

import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_variables', )
    args = parser.parse_args()

    num_vars = args.num_variables
    value_vec_length = int(2 ** num_vars)
    inp_dir = f"res_n_{num_vars}"

    input_monom_id2str_path = os.path.join(inp_dir, "monom_id2str.txt")
    input_value_vector_str2monom_ids_path = os.path.join(inp_dir, "output_value_vector2monom_ids_path.txt")
    output_shortest_polys_path = os.path.join(inp_dir, "value_vector2poly_strings.txt")

    output_dir = os.path.dirname(output_shortest_polys_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    monom_id2str = read_monom_id2str(input_monom_id2str_path)
    func_id2monom_ids = read_shortest_poly_monom_ids(input_value_vector_str2monom_ids_path)
    min_polys = []
    for func_id, monom_ids in func_id2monom_ids.items():
        func_value_vec = str(func_id)
        func_value_vec = '0' * (max(value_vec_length - len(func_value_vec), 0)) + func_value_vec
        # func_value_vec = int2binary_string(func_id, value_vec_length)
        monom_strs = [monom_id2str[m_id] for m_id in monom_ids]
        monom_strs.sort()
        poly_str = ' + '.join(monom_strs)

        min_polys.append((func_value_vec, poly_str))
    min_polys.sort(key=lambda t: t[0])
    with codecs.open(output_shortest_polys_path, 'w+', encoding="utf-8") as out_file:
        for (func_val_vec, poly_str) in min_polys:
            out_file.write(f"{func_val_vec}\t{poly_str}\n")



if __name__ == '__main__':
    main()
