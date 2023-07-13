"""
Coded by Charles Assaad
"""

import networkx as nx


def remove_self_loops(sg):
    """
    Transform (remove self loops)
    :param sg: a summary causal graph
    :return:
    """
    dag = sg.copy()
    edges = list(dag.edges)
    for edge in edges:
        if edge[0] == edge[1]:
            dag.remove_edge(edge[0], edge[0])
    return dag


def adjutment_set_for_direct_effect_in_ascgl(ascgl, x, y, gamma_max=1, gamma_min_dict=None):
    dag = remove_self_loops(ascgl)
    temporal_adjutment_set_for_each_gamma_xy = dict()
    adjutment_set = []
    if x in dag.predecessors(y):
        g_manip = dag.copy()
        g_manip.remove_edge(x, y)
        par_y = list(g_manip.predecessors(y))
        par_x = list(g_manip.predecessors(x))
        adjutment_set = par_x + par_y

        # get temporal vertices
        gamma_min_xy = gamma_min_dict[(x, y)]
        par_y_temporal_dict = dict()
        par_x_temporal_dict = dict()
        for gamma_xy in range(gamma_min_xy, gamma_max + 1):
            par_y_temporal_dict[gamma_xy] = []
            par_x_temporal_dict[gamma_xy] = []
            for b in par_x:
                min_gamma_bx = gamma_min_dict[(b, x)]
                for gamma in list(range(gamma_xy + min_gamma_bx, gamma_xy + gamma_max + 1)):
                    bt = str(b) + "_t"
                    if gamma > gamma_xy:
                        bt = bt + "_" + str(gamma)
                    par_x_temporal_dict[gamma_xy].append(bt)

            for s in par_y:
                min_gamma_sy = gamma_min_dict[(s, y)]
                for gamma in list(range(min_gamma_sy, gamma_max + 1)):
                    st = str(s) + "_t"
                    if gamma > 0:
                        st = st + "_" + str(gamma)
                    par_y_temporal_dict[gamma_xy].append(st)

            if ((x, x) in ascgl.edges) and ((y, y) in ascgl.edges):
                for gamma in list(range(1, gamma_xy + 1)):
                    st = str(y) + "_t_" + str(gamma)
                    par_y_temporal_dict[gamma_xy].append(st)

                for gamma in list(range(gamma_min_xy, gamma_xy)):
                    st = str(x) + "_t"
                    if gamma > 0:
                        st = str(x) + "_t_" + str(gamma)
                    par_y_temporal_dict[gamma_xy].append(st)

                for gamma in list(range(gamma_xy + 1, gamma_xy + gamma_max + 1)):
                    bt = str(x) + "_t_" + str(gamma)
                    par_x_temporal_dict[gamma_xy].append(bt)

            temporal_adjutment_set_for_each_gamma_xy[gamma_xy] = par_x_temporal_dict[gamma_xy] + par_y_temporal_dict[gamma_xy]

    return temporal_adjutment_set_for_each_gamma_xy, adjutment_set


def backdoor_from_summary_graph(sg, x, y, gamma_max=1, gamma_min_dict=None, xy_d_sep_by_empty_in_manip_graph=True):
    """
        sd
    :param sg: a summary causal graph
    :return:
    """
    if x == y:
        print("Error: x and y should be different")
        exit(0)
    if (x not in sg.nodes) or (y not in sg.nodes):
        print("Error: x and y should be in the graph")
        exit(0)
    if gamma_min_dict is None:
        gamma_min_dict = dict()
        for edge in sg.edges:
            gamma_min_dict[edge] = 0
    else:
        for edge in sg.edges:
            if gamma_min_dict[edge] > gamma_max:
                print("Error: One minimum lag (in gamma_min_dict) is greater than the maximum lag (gamma_max)")
                exit(0)

    dag = remove_self_loops(sg)

    backdoor_set = []
    if x in nx.ancestors(dag, y):
        g_manip = dag.copy()
        all_paths_x_to_y = nx.all_simple_paths(dag, source=x, target=y)
        list_deleted_edges = []
        for path in all_paths_x_to_y:
            if (path[0], path[1]) not in list_deleted_edges:
                g_manip.remove_edge(path[0], path[1])
                list_deleted_edges.append((path[0], path[1]))
        ancestor_of_y_in_manip = nx.ancestors(g_manip, y)

        # list of parents of x
        par_x = list(g_manip.predecessors(x))
        for p in par_x:
            if p in ancestor_of_y_in_manip:
                all_paths_p_to_y = list(nx.all_simple_paths(g_manip, source=p, target=y))
                nb_irrelevant_paths = 0
                for path in all_paths_p_to_y:
                    if len(list(set(path) & set(par_x))) > 1:
                        nb_irrelevant_paths = nb_irrelevant_paths + 1
                if nb_irrelevant_paths < len(all_paths_p_to_y):
                    backdoor_set.append(p)
            else:
                ancestor_of_p_in_manip = nx.ancestors(g_manip, p)
                common_anc = list(set(ancestor_of_y_in_manip) & set(ancestor_of_p_in_manip))
                if len(common_anc) > 0:
                    nb_irrelevant_paths_to_p = 0
                    nb_paths_to_p = 0
                    for a in common_anc:
                        all_paths_a_to_p = list(nx.all_simple_paths(g_manip, source=a, target=p))
                        nb_paths_to_p = nb_paths_to_p + len(all_paths_a_to_p)
                        for path in all_paths_a_to_p:
                            if len(list(set(path) & set(par_x))) > 1:
                                nb_irrelevant_paths_to_p = nb_irrelevant_paths_to_p + 1
                    if nb_irrelevant_paths_to_p < nb_paths_to_p:
                        backdoor_set.append(p)

    # if xy_d_sep_by_empty_in_manip_graph:
    if True:
        # gamma_min_xy = gamma_min_dict[(x, y)]
        # backdoor_temporal_dict = dict()
        # backdoor_temporal_dict[gamma_min_xy] = []
        # for b in backdoor_set:
        #     min_gamma_bx = gamma_min_dict[(b, x)]
        #     for gamma in list(range(gamma_min_xy + min_gamma_bx, gamma_min_xy + gamma_max + 1)):
        #         bt = str(b) + "_t"
        #         if gamma > gamma_min_xy:
        #             bt = bt + "_" + str(gamma)
        #         backdoor_temporal_dict[gamma_min_xy].append(bt)
        #
        # if ((x, x) in sg.edges) and ((y, y) in sg.edges):
        #     for gamma in list(range(gamma_min_xy + 1, gamma_min_xy + gamma_max + 1)):
        #         bt = str(x) + "_t_" + str(gamma)
        #         backdoor_temporal_dict[gamma_min_xy].append(bt)


        gamma_min_xy = gamma_min_dict[(x, y)]
        backdoor_temporal_dict = dict()
        for gamma_xy in range(gamma_min_xy, gamma_max + 1):
            backdoor_temporal_dict[gamma_xy] = []
            for b in backdoor_set:
                min_gamma_bx = gamma_min_dict[(b, x)]
                for gamma in list(range(gamma_xy + min_gamma_bx, gamma_xy + gamma_max + 1)):
                    bt = str(b) + "_t"
                    if gamma > gamma_xy:
                        bt = bt + "_" + str(gamma)
                    backdoor_temporal_dict[gamma_xy].append(bt)

            if ((x, x) in sg.edges) and ((y, y) in sg.edges):
                for gamma in list(range(gamma_xy + 1, gamma_xy + gamma_max + 1)):
                    bt = str(x) + "_t_" + str(gamma)
                    backdoor_temporal_dict[gamma_xy].append(bt)


    # else:
    #     backdoor_temporal_dict = dict()
    #     for gamma_min_xy in range(gamma_min_dict[(x, y)], gamma_max + 1):
    #         backdoor_temporal_dict[gamma_min_xy] = []
    #         for b in backdoor_set:
    #             min_gamma_bx = gamma_min_dict[(b, x)]
    #             for gamma in list(range(gamma_min_xy + min_gamma_bx, gamma_min_xy + gamma_max + 1)):
    #                 bt = str(b) + "_t"
    #                 if gamma > gamma_min_xy:
    #                     bt = bt + "_" + str(gamma)
    #                 backdoor_temporal_dict[gamma_min_xy].append(bt)
    #
    #         if ((x, x) in sg.edges) and ((y, y) in sg.edges):
    #             for gamma in list(range(gamma_min_xy + 1, gamma_min_xy + gamma_max + 1)):
    #                 bt = str(x) + "_t_" + str(gamma)
    #                 backdoor_temporal_dict[gamma_min_xy].append(bt)

    return backdoor_temporal_dict, backdoor_set



def singledoor_from_summary_graph(sg, x, y, gamma_max=1, gamma_min_dict=None, xy_d_sep_by_empty_in_manip_graph=True):
    dag = remove_self_loops(sg)
    singledoor_set = []
    if x in nx.ancestors(dag, y):
        g_manip = dag.copy()
        g_manip.remove_edge(x, y)
        par_y = list(g_manip.predecessors(y))
        for p in par_y:
            all_paths_x_to_p = list(nx.all_simple_paths(g_manip, source=x, target=p))
            nb_irrelevant_paths = 0
            for path in all_paths_x_to_p:
                if len(list(set(path) & set(par_y))) > 1:
                    nb_irrelevant_paths = nb_irrelevant_paths + 1
            if nb_irrelevant_paths < len(all_paths_x_to_p):
                singledoor_set.append(p)

    gamma_min_xy = gamma_min_dict[(x, y)]
    singledoor_temporal_dict = dict()
    for gamma_xy in range(gamma_min_xy, gamma_max + 1):
        singledoor_temporal_dict[gamma_xy] = []
        for s in singledoor_set:
            min_gamma_sy = gamma_min_dict[(s, y)]
            for gamma in list(range(min_gamma_sy, gamma_max + 1)):
                st = str(s) + "_t"
                if gamma > 0:
                    st = st + "_" + str(gamma)
                singledoor_temporal_dict[gamma_xy].append(st)

        if ((x, x) in sg.edges) and ((y, y) in sg.edges):
            for gamma in list(range(1, gamma_xy + 1)):
                st = str(y) + "_t_" + str(gamma)
                singledoor_temporal_dict[gamma_xy].append(st)

            for gamma in list(range(gamma_min_xy, gamma_xy)):
                st = str(x) + "_t"
                if gamma > 0:
                    st = str(x) + "_t_" + str(gamma)
                singledoor_temporal_dict[gamma_xy].append(st)

    return singledoor_temporal_dict, singledoor_set


if __name__ == '__main__':
    param_structure = "big"
    if param_structure == "v_structure":
        G = nx.DiGraph()
        G.add_nodes_from(["V2", "V3", "V4"])
        G.add_edge("V2", "V2")
        G.add_edge("V3", "V3")
        G.add_edge("V2", "V4")
        G.add_edge("V3", "V4")
    elif param_structure == "chain":
        G = nx.DiGraph()
        G.add_nodes_from(["V2", "V3", "V4"])
        G.add_edge("V2", "V2")
        G.add_edge("V3", "V3")
        G.add_edge("V2", "V3")
        G.add_edge("V3", "V4")
    elif param_structure == "diamond":
        G = nx.DiGraph()
        G.add_nodes_from(["V1", "V2", "V3", "V4"])
        G.add_edge("V1", "V1")
        G.add_edge("V2", "V2")
        G.add_edge("V3", "V3")
        G.add_edge("V1", "V2")
        G.add_edge("V1", "V3")
        G.add_edge("V2", "V4")
        G.add_edge("V3", "V4")
    elif param_structure == "big":
        G = nx.DiGraph()
        G.add_nodes_from(["V1", "V2", "V3", "V4"])
        G.add_edge("V1", "V1")
        G.add_edge("V2", "V2")
        G.add_edge("V3", "V3")
        G.add_edge("V4", "V4")
        G.add_edge("V5", "V5")
        G.add_edge("V6", "V6")

        G.add_edge("V1", "V2")
        G.add_edge("V2", "V3")
        G.add_edge("V1", "V4")
        G.add_edge("V4", "V5")
        G.add_edge("V5", "V6")
        G.add_edge("V3", "V6")

        G.add_edge("V0", "V1")
        G.add_edge("V0", "V4")
        G.add_edge("V4", "Va")
        G.add_edge("Va", "V5")

        G.add_edge("V4", "V6")


    dd = dict()
    for edge in G.edges:
        dd[edge] = 1
    res, b_set = backdoor_from_summary_graph(G, "V4", "V6", gamma_max=3, gamma_min_dict=dd)
    print(res)
    res = singledoor_from_summary_graph(G, "V4", "V6", gamma_max=3, gamma_min_dict=dd)
    print(res)

