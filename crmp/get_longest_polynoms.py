import os

from generalized_boolean_polynoms.practice_sem_3.utils import read_shortest_poly_monom_ids, read_monom_id2str, \
    get_longest_poly_monom_ids, int2binary_string


def main():
    num_vars = 3
    val_vec_length = 2 ** num_vars
    input_dir = f"res_n_{num_vars}_new"

    input_monom_id2str_path = os.path.join(input_dir, "monom_id2str.txt")
    input_shortest_poly_ids_path = os.path.join(input_dir, "output_value_vector2monom_ids_path.txt")

    monom_id2str = read_monom_id2str(input_monom_id2str_path)
    minimum_poly2monom_ids = read_shortest_poly_monom_ids(poly_id2monom_ids_path=input_shortest_poly_ids_path)

    longest_polys = get_longest_poly_monom_ids(minimum_poly2monom_ids=minimum_poly2monom_ids)
    print(f"Num longest function = {len(longest_polys)}")

    for function_id, monom_id in longest_polys.items():
        func_val_vec_str = int2binary_string(integer=function_id, length=val_vec_length)
        monom_strs = [monom_id2str[m_id] for m_id in monom_id]
        print(f"{func_val_vec_str}\t{' + '.join(monom_strs)}")
    print(f"Num longest function = {len(longest_polys)}")




if __name__ == '__main__':
    main()