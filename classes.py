from collections import Counter

from generalized_boolean_polynoms.utils import LITERALS, TRANSFORMATIONS_VERBOSE


class Polynom:
    def __init__(self, monoms, ):
        monoms = [tuple(t) for t in monoms]
        monom_counter = Counter(monoms)
        filtered_monoms = [monom for monom, count in monom_counter.items() if count % 2 == 1]
        filtered_monoms.sort()
        self.monoms = filtered_monoms

    def filter_monoms(self):
        self.monoms = [tuple(t) for t in self.monoms]
        monom_counter = Counter(self.monoms)
        self.monoms = [monom for monom, count in monom_counter.items() if count % 2 == 1]

    def sort_monoms(self):
        self.monoms.sort()

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

    def __len__(self):
        return len(self.monoms)

    def __mul__(self, other):
        if other == 1:
            return self
        elif isinstance(other, Polynom):
            new_poly = Polynom(list())
            for monom_1 in self.monoms:
                for monom_2 in other.monoms:
                    num_literals = len(monom_1)
                    new_mask = [0, ] * num_literals
                    reduced_monom = False
                    for literal_id in range(num_literals):
                        lit_1 = monom_1[literal_id]
                        lit_2 = monom_2[literal_id]
                        if lit_1 == lit_2:
                            #  (-1, -1), (0, 0), (1, 1)
                            new_lit = lit_1
                        elif lit_1 == 0:
                            # (0, -1), (0, 1),
                            new_lit = lit_2
                        elif lit_2 == 0:
                            # (-1, 0), (1, 0)
                            new_lit = lit_1
                        else:
                            # (-1, 1), (1, -1),
                            new_lit = None
                            reduced_monom = True
                        if reduced_monom:
                            break
                        else:
                            new_mask[literal_id] = new_lit
                    if reduced_monom:

                        continue
                    else:
                        new_poly.monoms.append(new_mask)
            if len(self.monoms) == 0:
                new_poly.monoms = self.monoms
            elif len(other.monoms) == 0:
                new_poly.monoms = other.monoms
            new_poly.filter_monoms()
            new_poly.sort_monoms()
            return new_poly
        else:
            raise ValueError(f"Polynom multiplication error")


class Transformation:

    def __init__(self, source_poly: Polynom, dest_poly: Polynom, transform_type, literal_id, processed_monom_mask):
        self.source_poly = source_poly
        self.dest_poly = dest_poly
        self.transform_type = transform_type
        self.literal_id = literal_id
        self.processed_monom_mask = processed_monom_mask

    def __str__(self):
        return f"[{self.source_poly}] --- {TRANSFORMATIONS_VERBOSE[self.transform_type]} ---> [{self.dest_poly}]"

    def __repr__(self):
        return self.__str__()
