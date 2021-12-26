from generalized_boolean_polynoms.classes import Polynom
from generalized_boolean_polynoms.utils import get_monoms_literal_val_monom_literal_id


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

    negative_monom_id, negative_lit_id = get_monoms_literal_val_monom_literal_id(gen_poly_monoms, -1)
    while (negative_monom_id, negative_lit_id) != (-1, -1):
        monom_mask = gen_poly_monoms[negative_monom_id]
        new_monoms, _ = apply_transformation_to_monom(monom_mask, negative_lit_id, )
        gen_poly_monoms.remove(monom_mask)
        gen_poly_monoms.extend(new_monoms)
        zhegalkin_poly.monoms = gen_poly_monoms
        zhegalkin_poly.filter_monoms()
        gen_poly_monoms = zhegalkin_poly.monoms
        negative_monom_id, negative_lit_id = get_monoms_literal_val_monom_literal_id(gen_poly_monoms, -1)
    zhegalkin_poly.sort_monoms()

    return zhegalkin_poly


def check_polynom_has_non_expanding_transform(poly: Polynom, num_literals: int):
    polynom_monoms = poly.monoms
    for monom_mask in polynom_monoms:
        num_literals = len(monom_mask)
        for literal_id in range(num_literals):
            # Заводим копию полинома
            new_polynom_monoms = list(poly.monoms)
            new_polynom_monoms.remove(monom_mask)
            new_monoms, transform_type = apply_transformation_to_monom(monom_mask, literal_id, )
            # Добавляем новые мономы в список полиномов нового монома
            new_polynom_monoms.extend(new_monoms)
            # Создаём новый полином по результатам преобразования
            new_polynom = Polynom(new_polynom_monoms)
            if len(new_polynom) < len(poly):
                return True
    # Добавляем представление нуля
    for literal_id in range(num_literals):
        # Заводим копию полинома
        new_polynom_monoms = list(poly.monoms)
        new_monoms, transform_type = apply_transformation_to_monom(monom=None, literal_id=literal_id,
                                                                   num_literals=num_literals)
        new_polynom_monoms.extend(new_monoms)
        # Создаём новый полином по результатам преобразования
        new_polynom = Polynom(new_polynom_monoms)
        if len(new_polynom) < len(poly):
            return True

    return False
