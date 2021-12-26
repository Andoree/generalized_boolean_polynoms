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
