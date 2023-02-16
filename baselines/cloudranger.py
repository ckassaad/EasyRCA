import pandas as pd
import numpy as np
import networkx as nx

from causallearn.search.ConstraintBased.PC import pc
import matplotlib.pyplot as plt

from tigramite.independence_tests import ParCorr
from estimation import FisherZ
import random

def find_transition_matrix(g, df, backward_step=0.1):
    transition_df = pd.DataFrame(np.zeros([len(df.columns), len(df.columns)]), columns=df.columns, index=df.columns)
    for edge in g.edges:
        cause = edge[0]
        effect = edge[1]
        corr = FisherZ(cause, effect)
        stat = abs(corr.get_dependence(df))
        transition_df[effect].loc[cause] = stat
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
                    corr = FisherZ(node, col_k)
                    stat = abs(corr.get_dependence(df))
                    P_pc_max.append(stat)
            if len(P_pc_max) > 0:
                P_pc_max = np.max(P_pc_max)
            else:
                P_pc_max = 0

            if node != col_i:
                corr = FisherZ(node, col_i)
                stat = abs(corr.get_dependence(df))
                q_ii = stat
                if q_ii > P_pc_max:
                    if q_ii - P_pc_max > transition_df.loc[col_i][col_i]:
                        transition_df.loc[col_i][col_i] = q_ii - P_pc_max
        # transition_df[node].loc[node] = 0.1

    #normalizing
    for effect in g.nodes:
        total_corr = sum(transition_df[effect])
        if total_corr > 0:
            for node in g.nodes:
                transition_df[effect].loc[node] = transition_df[effect].loc[node]/total_corr
    print(transition_df)
    return transition_df


def random_walk(g, transition_df, walkLength=10):
    transition_matrix = transition_df.values
    p = np.zeros([len(g.nodes)]).reshape(-1, 1)
    p[0, 0] = 1
    # visited = list()
    # print(transition_df)
    # print(p)
    # for k in range(walkLength):
    #     # evaluate the next state vector
    #     p = np.dot(transition_matrix, p)
    #     # choose the node with higher probability as the visited node
    #     visited.append(transition_df.columns[np.argmax(p)])

    i = random.randint(0, len(g.nodes)-1)
    visited = [transition_df.columns[i]]
    I = np.arange(len(p))
    for _ in range(walkLength):
        i = np.random.choice(I, p=transition_matrix[:,i])
        visited.append(transition_df.columns[i])
    return visited


def cloud_ranger(data, anomalous_nodes, anomalies_start_time=None, anomaly_length=200, sig_threshold=0.05):

    last_start_time_anomaly = 0
    for node in anomalous_nodes:
        last_start_time_anomaly = max(last_start_time_anomaly, anomalies_start_time[node])
    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]

    output = pc(anomalous_data.values, sig_threshold, "fisherz", verbose=True)
    # output.G.graph[4, 5] = 1

    g = nx.DiGraph()
    g.add_nodes_from(data.columns)
    for c in range(output.G.graph.shape[1]):
        for e in range(output.G.graph.shape[0]):
            if output.G.graph[e, c] == 1:
                g.add_edge(data.columns[c], data.columns[e])
            elif (output.G.graph[e, c] == -1) and (output.G.graph[c, e] == -1):
                g.add_edge(data.columns[c], data.columns[e])
                g.add_edge(data.columns[e], data.columns[c])

    # print(output.draw_pydot_graph())

    # Random walk
    transition_df = find_transition_matrix(g, data, backward_step=0.1)
    visited = random_walk(g, transition_df, walkLength=1000)
    print(visited)

    freq_dict = {x: visited.count(x) for x in visited}

    rc_list = list()
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
    #
    #     print(res_dict)
    #     roots = [r for r in anomalous_nodes if r not in not_roots]
    #     return roots


def cloud_ranger0(graph, data, anomalous_nodes, anomalies_start_time=None, anomaly_length=200, sig_threshold=0.05):

    last_start_time_anomaly = 0
    for node in anomalous_nodes:
        last_start_time_anomaly = max(last_start_time_anomaly, anomalies_start_time[node])
    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]

    g = graph
    # print(output.draw_pydot_graph())

    # Random walk
    transition_df = find_transition_matrix(g, data, backward_step=0.1)
    visited = random_walk(g, transition_df, walkLength=1000)
    print(visited)

    freq_dict = {x: visited.count(x) for x in visited}

    rc_list = list()
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
        anomalies_start[ano] = 5500
    anomaly_length = 1000

    a = np.random.normal(size=10000)
    b = 2*a + 0.1 * np.random.normal(size=10000)
    c = 5*a + 0.2 * np.random.normal(size=10000)
    d = 5*b + 0.2 * np.random.normal(size=10000)
    e = 5*c + 0.2 * np.random.normal(size=10000)
    f = 5 * d + 5 * e + 0.2 * np.random.normal(size=10000)

    y = np.random.normal(1, 1, size=10000)
    z = 3 * y + 0.2 * np.random.normal(size=10000)
    data = pd.DataFrame(np.array([a, b, c, d, e, f, y, z]).T, columns=["a", "b", "c", "d", "e", "f", "y", "z"])

    ad = np.array(4 * e[anomalies_start["f"]: anomalies_start["f"] + anomaly_length] + 0.2 * np.random.normal(size=anomaly_length))
    print(ad.shape)
    data["f"].loc[anomalies_start["f"]: anomalies_start["f"] + anomaly_length-1] = ad

    # data["f"].loc[anomalies_start["f"]: anomalies_start["f"] + anomaly_length-1] = np.random.normal(size=anomaly_length)


    res = cloud_ranger(data, anomalous, anomalies_start_time=anomalies_start, anomaly_length=anomaly_length)
    print(res)
