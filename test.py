a = [(1, 2), (2, 3), (4 , 5)]
b = [(1, 2), (2, 3), (4 , 5)]
c = [(1, 2), (2, 3), ]
print(a == b)
print(a == c)

# s = f"Polynom {node_index[vertex_id]}, length = {len(reversed_path)}: "
                # s_tex = f"Polynom ${polynom_str_to_tex(node_index[vertex_id])}$, Number of transformations = {len(reversed_path) - 1}\n\\begin{'{' + 'dmath' + '}'}\n"
                #
                # for i in range(len(reversed_path) - 1):
                #     polynom_source_id = reversed_path[i]
                #     polynom_dest_id = reversed_path[i + 1]
                #     transformation_entry = edges_df[
                #         (edges_df["poly_1_id"] == polynom_source_id) & (edges_df["poly_2_id"] == polynom_dest_id)]
                #
                #     transform_type_id = transformation_entry["transform_type_id"].values[0]
                #     transform_edge_literal_id = transformation_entry["literal_id"].values[0]
                #     transform_edge_monom = transformation_entry["target_monom_mask"].values[0]
                #
                #     transform_verbose_mask = TRANSFORMATIONS_VERBOSE_MASKS[transform_type_id]
                #     transform_verbose_tex_mask = TRANSFORMATIONS_VERBOSE_TEX_MASKS[transform_type_id]
                #     transform_edge_literal = LITERALS[transform_edge_literal_id]
                #     transform_verbose = transform_verbose_mask.replace("<literal>", transform_edge_literal)
                #     transform_verbose_tex = transform_verbose_tex_mask.replace("<literal>", transform_edge_literal)
                #     transform_edge_monom_str = monom_mask_to_str(transform_edge_monom)
                #     transform_edge_monom_tex_str = monom_mask_to_tex_str(transform_edge_monom)
                #
                #     # transformation_type_verbose = TRANSFORMATIONS_VERBOSE[transform_type_id]
                #     polynom_source_verbose = node_index[polynom_source_id]
                #     polynom_source_verbose_tex = polynom_str_to_tex(polynom_source_verbose)
                #
                #     s += f"[{polynom_source_verbose}] = [apply {transform_verbose} to {transform_edge_monom_str}] = "
                #     if len(polynom_source_verbose_tex) > 60:
                #         first_half_tex, second_half_tex = split_polynom_str(polynom_source_verbose_tex)
                #         s_tex += fr"{'{'}{first_half_tex} + {'}'} + "
                #         s_tex += fr"{'{'}{second_half_tex} = [Apply\,({transform_verbose_tex})\,\,to\,\,{transform_edge_monom_tex_str}]{'}'} = "
                #     else:
                #         s_tex += fr"{'{'}{polynom_source_verbose_tex} = [Apply\,({transform_verbose_tex})\,\,to\,\,{transform_edge_monom_tex_str}]{'}'} = "
                # if len(reversed_path) > 1:
                #     polynom_dest_verbose = node_index[polynom_dest_id]
                #     polynom_dest_verbose_tex = polynom_str_to_tex(polynom_dest_verbose)
                #     s += polynom_dest_verbose
                #     s_tex += polynom_dest_verbose_tex
                # s_tex += "\n\\end{dmath}"