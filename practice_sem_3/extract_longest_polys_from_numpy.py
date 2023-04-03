import codecs
import os.path

from generalized_boolean_polynoms.practice_sem_3.utils import read_monom_id2str, load_poly_id2min_poly_monom_ids_numpy, \
    int2binary_string


def main():
    num_vars = 4
    numpy_poly_npy_path = f"res_new/res_n_{num_vars}_new/numpy_polys/0.npy"
    input_dir = f"res_new/res_n_{num_vars}_new/"

    monom2id_path = os.path.join(input_dir, "monom_id2str.txt")
    longest_val_vec_str2min_poly_path = os.path.join(input_dir, "longest_val_vec_str2min_poly.txt")
    val_vec_str2min_poly_path = os.path.join(input_dir, "val_vec_str2min_poly.txt")


    monom_id2str = read_monom_id2str(monom_id2str_path=monom2id_path)
    max_monom_id = max(monom_id2str.keys())

    p_id2min_p_monom_ids = load_poly_id2min_poly_monom_ids_numpy(npy_file_path=numpy_poly_npy_path,
                                                                 max_monom_id=max_monom_id)
    max_poly_length = max(len(t) for t in p_id2min_p_monom_ids.values())
    with codecs.open(longest_val_vec_str2min_poly_path, 'w+', encoding="utf-8") as out_file_1, \
            codecs.open(val_vec_str2min_poly_path, 'w+', encoding="utf-8") as out_file_2:
        for f_id, m_ids in p_id2min_p_monom_ids.items():
            f_val_vec = int2binary_string(integer=f_id, length=2 ** num_vars)
            poly_str = ' + '.join(monom_id2str[m_i] for m_i in m_ids)
            if len(m_ids) == max_poly_length:
                print(f_val_vec, poly_str)
                out_file_1.write(f"{f_val_vec}\t{poly_str}\n")
            out_file_2.write(f"{f_val_vec}\t{poly_str}\n")


if __name__ == '__main__':
    main()
