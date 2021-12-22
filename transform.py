from generalized_boolean_polynoms.create_polynoms_graph import Polynom


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

def generalized_polynom_to_zhegalkin(gen_poly: Polynom):
    gen_poly_monoms = list(gen_poly.monoms)
    zhegalkin_poly = Polynom(gen_poly_monoms)
    """
    1. Прохожусь по всем  мономам, если вижу моном с отрицанием, то:
    2. Применяю преобразование
    3. Обновить полином: удалить старый моном и добавить новые, полученные после преобразования?
    4. Удалить дубликаты мономов, отсортировать
    """
    has_negative_literals = True
    while has_negative_literals:
        for monom_mask in gen_poly_monoms:
            num_literals = len(monom_mask)
            monom_has_changed = False
            for literal_id in range(num_literals):
                literal_value = monom_mask[literal_id]
                assert literal_value in (-1, 0, 1)
                if literal_value == -1:
                    gen_poly_monoms.remove(monom_mask)
                    new_monoms, _ = apply_transformation_to_monom(monom_mask, literal_id, )
                    gen_poly_monoms.extend(new_monoms)
                    zhegalkin_poly.monoms = gen_poly_monoms
                    zhegalkin_poly.filter_monoms()
                    monom_has_changed = True
                    break
            if monom_has_changed:
                break
        has_negative_literals = False
    return zhegalkin_poly

