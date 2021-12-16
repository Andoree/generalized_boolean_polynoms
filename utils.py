LITERALS = ['x', 'y', 'z']
TRANSFORMATIONS_VERBOSE = {
    1: "x = -x + 1",
    2: "0 = x +(-x) + 1",
    0: "1 = x + -x",
    -1: "-x = x + 1"
}
TRANSFORMATIONS_VERBOSE_MASKS = {
    1: "<literal> = -<literal> + 1",
    2: "0 = <literal> +(-<literal>) + 1",
    0: "1 = <literal> + -<literal>",
    -1: "-<literal> = <literal> + 1"
}
