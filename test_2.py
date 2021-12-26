
while True:
    for i in range(10):
        has_changed = False
        for j in range(10):
            print(i, j)
            has_changed = True
            break
        if has_changed:
            break



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