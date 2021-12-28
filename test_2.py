#
# while True:
#     for i in range(10):
#         has_changed = False
#         for j in range(10):
#             print(i, j)
#             has_changed = True
#             break
#         if has_changed:
#             break


# while has_negative_literals:
# for monom_mask in gen_poly_monoms:
#     num_literals = len(monom_mask)
#     monom_has_changed = False
#     for literal_id in range(num_literals):
#         literal_value = monom_mask[literal_id]
#         assert literal_value in (-1, 0, 1)
#         if literal_value == -1:
#             gen_poly_monoms.remove(monom_mask)
#             new_monoms, _ = apply_transformation_to_monom(monom_mask, literal_id, )
#             gen_poly_monoms.extend(new_monoms)
#             zhegalkin_poly.monoms = gen_poly_monoms
#             zhegalkin_poly.filter_monoms()
#             monom_has_changed = True
#             break
#     if monom_has_changed:
#         break
# has_negative_literals = False

def main():
    s1 = 1 * ((1 / 2) ** 1)
    s2 = 8 * ((1 / 2) ** 2)
    s3 = 8 * 7 * ((1 / 2) ** 3)
    s4 = 8 * 7 * 6 * ((1 / 2) ** 4)
    s5 = 8 * 7 * 6 * 5 * ((1 / 2) ** 5)
    s6 = 8 * 7 * 6 * 5 * 4 * ((1 / 2) ** 6)
    s7 = 8 * 7 * 6 * 5 * 4 * 3 * ((1 / 2) ** 7)
    s8 = 8 * 7 * 6 * 5 * 4 * 3 * 2 * ((1 / 2) ** 8)
    s9 = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * ((1 / 2) ** 9)
    print(s1)
    print(s2)
    print(s3)
    print(s4)
    print(s5)
    print(s6)
    print(s7)
    print(s8)
    print(s9)
    print(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9)
    print(581 * 4 + 3)


if __name__ == '__main__':
    main()
