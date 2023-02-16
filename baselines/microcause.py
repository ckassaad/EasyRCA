import pandas as pd
import numpy as np
import networkx as nx

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from tigramite import data_processing as pp

from estimation import FisherZ
import random


def find_transition_matrix(g, df, backward_step=0.1):
    transition_df = pd.DataFrame(np.zeros([len(df.columns), len(df.columns)]), columns=df.columns, index=df.columns)
    for edge in g.edges:
        cause = edge[0]
        effect = edge[1]
        if cause != effect:
            # Forward step
            par_cause = list(g.predecessors(cause))
            par_effect = list(g.predecessors(effect))
            if cause in par_cause:
                par_cause.remove(cause)
            if effect in par_effect:
                par_effect.remove(effect)
            par_effect.remove(cause)
            cond_list = par_cause+par_effect
            cond_list = list(set(cond_list))
            corr = FisherZ(cause, effect, cond_list=cond_list)
            stat = abs(corr.get_dependence(df))
            print(cause, effect, cond_list, stat)
            # if type(stat) != int and type(stat) != float:
            #     stat = 0
            transition_df[effect].loc[cause] = stat
            # backward step
            transition_df[cause].loc[effect] = backward_step * stat

    # Self step
    adj = nx.to_numpy_matrix(g)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                adj[j, i] = 1
    for node in g.nodes:
        for i, col_i in enumerate(list(df.columns)):
            # Calculate P_pc^max
            P_pc_max = []
            # (k, i) in edges.
            for k in adj[:, i].nonzero()[0]:
                # Note for partial correlation, the two variables cannot be the same.
                col_k = df.columns[k]
                if node != col_k:
                    par_cause = list(g.predecessors(col_k))
                    par_effect = list(g.predecessors(node))
                    if col_k in par_cause:
                        par_cause.remove(col_k)
                    if node in par_effect:
                        par_effect.remove(node)
                    if col_k in par_effect:
                        par_effect.remove(col_k)
                    cond_list = par_cause + par_effect
                    cond_list = list(set(cond_list))
                    corr = FisherZ(node, col_k, cond_list=cond_list)
                    stat = abs(corr.get_dependence(df))
                    P_pc_max.append(stat)
            if len(P_pc_max) > 0:
                P_pc_max = np.max(P_pc_max)
            else:
                P_pc_max = 0

            if node != col_i:
                par_cause = list(g.predecessors(col_i))
                par_effect = list(g.predecessors(node))
                if col_i in par_cause:
                    par_cause.remove(col_i)
                if node in par_cause:
                    par_cause.remove(node)
                if node in par_effect:
                    par_effect.remove(node)
                if col_i in par_effect:
                    par_effect.remove(col_i)
                cond_list = par_cause + par_effect
                cond_list = list(set(cond_list))
                corr = FisherZ(node, col_i, cond_list=cond_list)
                stat = abs(corr.get_dependence(df))
                q_ii = stat
                if q_ii > P_pc_max:
                    if q_ii - P_pc_max > transition_df.loc[col_i][col_i]:
                        transition_df.loc[col_i][col_i] = q_ii - P_pc_max
        # transition_df[node].loc[node] = 0.01

    #normalizing
    print(transition_df)
    for effect in g.nodes:
        total_corr = sum(transition_df[effect])
        if total_corr > 0:
            for node in g.nodes:
                transition_df[effect].loc[node] = transition_df[effect].loc[node]/total_corr
    # for cause in g.nodes:
    #     total_corr = sum(transition_df.loc[cause])
    #     if total_corr > 0:
    #         for node in g.nodes:
    #             transition_df[node].loc[cause] = transition_df[node].loc[cause]/total_corr
    return transition_df


def random_walk(g, transition_df, start_idx=0, walkLength=10):
    transition_matrix = transition_df.values
    p = np.zeros([len(g.nodes)]).reshape(-1, 1)
    p[0, 0] = 1

    # visited = list()
    # I = np.arange(len(p))
    # for node in g.nodes:
    #     if (node, node) in g.edges:
    #         g.remove_edges_from([(node, node)])
    # try:
    #     orderingNodes = list(nx.topological_sort(g))
    #     i = orderingNodes.index(orderingNodes[-1])
    # except nx.exception.NetworkXUnfeasible:
    #     # i = random.randint(0, len(g.nodes) - 1)
    #     i = 0
    # # i = start_idx

    i = random.randint(0, len(g.nodes)-1)
    visited = [transition_df.columns[i]]
    I = np.arange(len(p))

    visited.append(transition_df.columns[i])
    for _ in range(walkLength):
        if sum(transition_matrix[:, i]) > 0.99:
            i = np.random.choice(I, p=transition_matrix[:, i])
        else:
            i = np.random.choice(I)
        visited.append(transition_df.columns[i])
    return visited


def micro_cause(data, anomalous_nodes, anomalies_start_time=None, anomaly_length=200, gamma_max=1, sig_threshold=0.05):

    last_start_time_anomaly = 0
    for node in anomalous_nodes:
        last_start_time_anomaly = max(last_start_time_anomaly, anomalies_start_time[node])
    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]

    cd_method = "PCMCI"
    if cd_method == "PCMCI":
        dataframe = pp.DataFrame(anomalous_data.values,
                                 datatime=np.arange(len(anomalous_data)),
                                 var_names=anomalous_data.columns)
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        output = pcmci.run_pcmciplus(tau_min=0, tau_max=gamma_max, pc_alpha=sig_threshold)
        g = nx.DiGraph()
        g.add_nodes_from(data.columns)
        # graph = output['graph']
        # sig_links = (graph != "") * (graph != "<--")
        # res_dict = dict()
        # for j in range(data.shape[1]):
        #     res_dict[pcmci.var_names[j]] = []
        #     links = {(p[0], -p[1]): np.abs(output['val_matrix'][p[0], j, abs(p[1])]) for p in
        #              zip(*np.where(sig_links[:, j, :]))}
        #     sorted_links = sorted(links, key=links.get, reverse=True)
        #     for p in sorted_links:
        #         res_dict[pcmci.var_names[j]].append((pcmci.var_names[p[0]], p[1]))
        #         g.add_edges_from([(pcmci.var_names[p[0]], pcmci.var_names[j])])

        # pcmci.print_significant_links(p_matrix=output['p_matrix'],
        #                                    val_matrix=output['val_matrix'],
        #                                    alpha_level=0.05)

        g = nx.DiGraph()
        g.add_nodes_from(data.columns)
        for name_y in pcmci.all_parents.keys():
            for name_x, t_xy in pcmci.all_parents[name_y]:
                if (data.columns[name_x], data.columns[name_y]) not in g.edges:
                    g.add_edges_from([(data.columns[name_x], data.columns[name_y])])

        transition_df = find_transition_matrix(g, data, backward_step=0.1)
        rc_list = list()
        # for i in range(len(data.columns)):
        visited = random_walk(g, transition_df, walkLength=1000)
        freq_dict = {x: visited.count(x) for x in visited}

        if len(freq_dict.keys()) > 1:
            for i in range(2):
                rc = max(freq_dict, key=freq_dict.get)
                if rc not in rc_list:
                    rc_list.append(rc)
                del freq_dict[rc]
        else:
            rc_list.append(list(freq_dict.keys())[0])
        return rc_list

        # not_roots = []
        # res_dict = dict()
        # graph = output['graph']
        # sig_links = (graph != "") * (graph != "<--")
        # for j in range(data.shape[1]):
        #     res_dict[obj.var_names[j]] = []
        #     links = {(p[0], -p[1]): np.abs(output['val_matrix'][p[0], j, abs(p[1])]) for p in
        #              zip(*np.where(sig_links[:, j, :]))}
        #     sorted_links = sorted(links, key=links.get, reverse=True)
        #     for p in sorted_links:
        #         res_dict[obj.var_names[j]].append((obj.var_names[p[0]], p[1]))
        #         not_roots.append(obj.var_names[p[0]])

        # print(res_dict)
        # roots = [r for r in anomalous_nodes if r not in not_roots]

        # return rc_list



def micro_cause0(graph, data, anomalous_nodes, anomalies_start_time=None, anomaly_length=200, gamma_max=1, sig_threshold=0.05):

    last_start_time_anomaly = 0
    for node in anomalous_nodes:
        last_start_time_anomaly = max(last_start_time_anomaly, anomalies_start_time[node])
    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]

    g = graph

    transition_df = find_transition_matrix(g, data, backward_step=0.1)
    rc_list = list()
    # for i in range(len(data.columns)):
    visited = random_walk(g, transition_df, walkLength=1000)
    freq_dict = {x: visited.count(x) for x in visited}

    if len(freq_dict.keys()) > 1:
        for i in range(2):
            rc = max(freq_dict, key=freq_dict.get)
            if rc not in rc_list:
                rc_list.append(rc)
            del freq_dict[rc]
    else:
        rc_list.append(list(freq_dict.keys())[0])
    return rc_list



if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_edges_from([("z", "y"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("d", "f"), ("e", "f")])
    anomalous = ["a", "b", "e", "f", "z", "y", "d"]
    anomalies_start = dict()
    for ano in anomalous:
        anomalies_start[ano] = 3500
    anomaly_length = 1000

    a = np.random.normal(size=10000)
    b = 2*a[1:] + 0.1 * np.random.normal(size=9999)
    c = 5*a[1:] + 0.2 * np.random.normal(size=9999)
    d = 5*b[1:] + 0.2 * np.random.normal(size=9998)
    e = 5*c[1:] + 0.2 * np.random.normal(size=9998)
    f = 5 * d[1:] + 5 * e[1:] + 0.2 * np.random.normal(size=9997)

    y = np.random.normal(1, 1, size=9998)
    z = 3 * y[1:] + 0.2 * np.random.normal(size=9997)
    data = pd.DataFrame(np.array([a[:-3], b[:-2], c[:-2], d[:-1], e[:-1], f, y[:-1], z]).T, columns=["a", "b", "c", "d", "e", "f", "y", "z"])


    ad = np.array(4 * e[anomalies_start["f"]: anomalies_start["f"] + anomaly_length] + 0.2 * np.random.normal(size=anomaly_length))
    data["f"].loc[anomalies_start["f"]: anomalies_start["f"] + anomaly_length-1] = ad

    # data["f"].loc[anomalies_start["f"]: anomalies_start["f"] + anomaly_length-1] = np.random.normal(size=anomaly_length)


    res = micro_cause(data, anomalous, anomalies_start_time=anomalies_start, anomaly_length=anomaly_length)
    print(res)

