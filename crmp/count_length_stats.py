import argparse
import codecs
from collections import Counter


def load_value_vector_str2minimum_poly_monom_ids(inp_path):
    value_vector_str2minimum_poly_monom_ids = {}
    with codecs.open(inp_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            val_vec_str = attrs[0]
            if len(attrs) < 2:
                value_vector_str2minimum_poly_monom_ids[val_vec_str] = []
                continue
            monom_ids = [int(x) for x in attrs[1].split(',')]
            value_vector_str2minimum_poly_monom_ids[val_vec_str] = monom_ids
    return value_vector_str2minimum_poly_monom_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_variables',)
    args = parser.parse_args()

    num_variables = args.num_variables

    input_minimum_poly_path = f"res_n_{num_variables}_new/output_value_vector2monom_ids_path.txt"
    output_minimum_poly_path = f"res_n_{num_variables}_new/new_length_stats.txt"

    value_vector_str2minimum_poly_monom_ids = load_value_vector_str2minimum_poly_monom_ids(input_minimum_poly_path)
    counter = Counter((len(t) for t in value_vector_str2minimum_poly_monom_ids.values()))

    with codecs.open(output_minimum_poly_path, 'w+', encoding="utf-8") as out_file:
        for length, count in counter.items():
            out_file.write(f"{length}\t{count}\n")




    pass


if __name__ == '__main__':
    main()