import networkx as nx
import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

from estimation import FisherZ

import matplotlib.pyplot as plt


def run_pcmci(data, pc_alpha = 0.05, gamma_max=3, verbosity=0):
    dataframe = pp.DataFrame(data.values,
                             datatime=np.arange(len(data)),
                             var_names=data.columns)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)
    pcmci_res = pcmci.run_pcmci(tau_max=gamma_max, pc_alpha=pc_alpha)

    pcmci_links = pcmci.return_significant_links(pcmci_res['p_matrix'], pcmci_res['val_matrix'], alpha_level=pc_alpha, include_lagzero_links=False)

    g = nx.DiGraph()
    for i in range(len(data.columns)):
        g.add_node(data.columns[i])
    for j, links in pcmci_links['link_dict'].items():
        for l in links:
            g.add_edge(data.columns[l[0]], data.columns[j])
    return g


def get_Q_matrix(g, data, rho=0.1):
    frontend = list(data.columns)

    corr = np.corrcoef(np.array(data).T)
    for i in range(corr.shape[0]):
        corr[i, i] = 0.0
    corr = np.abs(corr)

    Q = np.zeros([len(data.columns), len(data.columns)])
    for e in g.edges():
        Q[e[0], e[1]] = corr[frontend[0] - 1, e[1]]
        backward_e = (e[1], e[0])
        if backward_e not in g.edges():
            Q[e[1], e[0]] = rho * corr[frontend[0] - 1, e[0]]

    adj = nx.adj_matrix(g).todense()
    for i in range(len(data.columns)):
        P_pc_max = None
        res_l = np.array([corr[frontend[0] - 1, k] for k in adj[:, i]])
        if corr[frontend[0] - 1, i] > np.max(res_l):
            Q[i, i] = corr[frontend[0] - 1, i] - np.max(res_l)
        else:
            Q[i, i] = 0
    l = []
    for i in np.sum(Q, axis=1):
        if i > 0:
            l.append(1.0 / i)
        else:
            l.append(0.0)
    l = np.diag(l)
    Q = np.dot(l, Q)
    return Q


def randomwalk(P, epochs, start_node, names, walk_step=50, print_trace=False):
    n = P.shape[0]
    score = dict()
    for node in names:
        score[node] = 0
    for epoch in range(epochs):
        current = start_node
        if print_trace:
            print("\n{:2d}".format(current), end="->")
        for step in range(walk_step):
            if np.sum(P.loc[current]) == 0:
                break
            else:
                next_node = names[np.random.choice(range(n), p=P.loc[current])]
                if print_trace:
                    print("{:2d}".format(current), end="->")
                score[next_node] += 1
                current = next_node
    # label = [i for i in range(n)]
    # score_list = list(zip(label, score))
    # score_list.sort(key=lambda x: x[1], reverse=True)
    return score


def get_Q_matrix_part_corr(data, g, target_node, rho=0.1):
    frontend = list(data.columns)

    df = data

    def get_part_corr(x, y, g):
        par_cause = list(g.predecessors(x))
        par_effect = list(g.predecessors(y))
        if x in par_cause:
            par_cause.remove(x)
        if y in par_effect:
            par_effect.remove(y)
        if x in par_effect:
            par_effect.remove(x)
        cond_list = par_cause + par_effect
        cond_list = list(set(cond_list))
        corr = FisherZ(x, y, cond_list=cond_list)
        ret = abs(corr.get_dependence(df))
        # For a valid transition probability, use absolute correlation values.
        return abs(float(ret))

    # Calculate the parent nodes set.
    pa_set = {}
    for e in g.edges():
        # Skip self links.
        if e[0] == e[1]:
            continue
        if e[1] not in pa_set:
            pa_set[e[1]] = set([e[0]])
        else:
            pa_set[e[1]].add(e[0])
    # Set an empty set for the nodes without parent nodes.
    for n in g.nodes():
        if n not in pa_set:
            pa_set[n] = set([])

    Q = pd.DataFrame(np.zeros([len(data.columns), len(data.columns)]), columns=data.columns, index=data.columns)
    for e in g.edges():
        # Do not add self links.
        if e[0] == e[1]:
            continue
        # e[0] --> e[1]: cause --> result
        # Forward step.
        # Note for partial correlation, the two variables cannot be the same.
        if target_node != e[0]:
            Q.loc[e[1]][e[0]] = get_part_corr(target_node, e[0], g)
        # Backward step
        backward_e = (e[1], e[0])
        # Note for partial correlation, the two variables cannot be the same.
        if backward_e not in g.edges() and target_node != e[1]:
            Q.loc[e[0]][e[1]] = rho * get_part_corr(target_node, e[1], g)

    adj = nx.to_numpy_matrix(g)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                adj[j, i] = 1
    # adj = nx.adj_matrix(g).todense()
    for i, col_i in enumerate(list(data.columns)):
        # Calculate P_pc^max
        P_pc_max = []
        # (k, i) in edges.
        for k in adj[:, i].nonzero()[0]:
            # Note for partial correlation, the two variables cannot be the same.
            col_k = data.columns[k]
            if target_node != col_k:
                P_pc_max.append(get_part_corr(target_node, col_k, g))
        if len(P_pc_max) > 0:
            P_pc_max = np.max(P_pc_max)
        else:
            P_pc_max = 0

        # Note for partial correlation, the two variables cannot be the same.
        if target_node != col_i:
            q_ii = get_part_corr(target_node, col_i, g)
            if q_ii > P_pc_max:
                Q.loc[col_i][col_i] = q_ii - P_pc_max
            else:
                Q.loc[col_i][col_i] = 0.01

    l = []
    for i in np.sum(Q.values, axis=1):
        if i > 0:
            l.append(1.0 / i)
        else:
            l.append(0.0)
    l = np.diag(l)
    Q = np.dot(l, Q)
    Q = pd.DataFrame(Q, columns=data.columns, index=data.columns)

    return Q


# def get_gamma(data, vis_list, eta=1000, lambda_param=0.8):
#     gamma = [0 for _ in range(len(data.columns))]
#     max_vis_time = np.max([vis_list[i] for i in vis_list.keys()])
# #     max_vis_time = 1.0
#     max_eta = np.max(eta)
#     for n in vis_list.keys():
#         vis = vis_list[n]
#         gamma[n] = lambda_param * vis / max_vis_time + (1-lambda_param) * eta[n] / max_eta
#     return gamma


def micro_cause2(data, anomalous_nodes, anomalies_start_time, anomaly_length, gamma_max=3, sig_threshold=0.05):
    last_start_time_anomaly = 0
    for node in anomalous_nodes:
        last_start_time_anomaly = max(last_start_time_anomaly, anomalies_start_time[node])
    first_end_time_anomaly = last_start_time_anomaly + anomaly_length
    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]
    data = anomalous_data

    g = run_pcmci(data, pc_alpha=sig_threshold, gamma_max=gamma_max, verbosity=1)

    # plt.figure(figsize=[12, 12])
    # nx.draw_networkx(g, pos=nx.circular_layout(g))
    # plt.show()

    vis_global = dict()
    for node in data.columns:
        vis_global[node] = 0
    for target_node in data.columns:
        Q = get_Q_matrix_part_corr(data, g, target_node, rho=0.1)
        # Q = get_Q_matrix_part_corr(g, rho=0.2)
        vis_list = randomwalk(Q, 10, target_node, names=data.columns, walk_step=1000)
        print(target_node, vis_list)
        for node in data.columns:
            if vis_global[node] < vis_list[node]:
                vis_global[node] = vis_list[node]

        # gamma = get_gamma(data, vis_list, lambda_param=0.5)
        # print(gamma)
    print(vis_global)
    rc_list = []
    if len(vis_global.keys()) > 1:
        for i in range(2):
            rc = max(vis_global, key=vis_global.get)
            if rc not in rc_list:
                rc_list.append(rc)
            del vis_global[rc]
    else:
        rc_list.append(list(vis_global.keys())[0])

    return rc_list


if __name__ == '__main__':
    import json
    from TestEasyRCA.generate_data import GenerateData

    nb_anomalous_data = 1000
    nb_data = 12 * nb_anomalous_data
    anomaly_start = nb_anomalous_data * 10
    list_graphs_interventions = []
    for di in [2, 3]:
        for do in [2, 3]:
            if (di == 1) or (di == 2) or (do == 2):
                for i in range(10):
                    list_graphs_interventions.append(str(di) + "_" + str(do) + "_" + str(i))

    str_i = list_graphs_interventions[0]
    with open('../TestEasyRCA/graphs_sim/degree_change/graphs/graph_' + str_i + '.json', 'r') as dict_file:
        dict_str = dict_file.read()
    dict_graph = json.loads(dict_str)
    file = open('../TestEasyRCA/graphs_sim/degree_change/interventions/intervention_' + str_i + '.csv', "r")
    intervention = file.read()
    file.close()
    print(intervention, dict_graph)

    graph = nx.DiGraph()
    graph.add_nodes_from(dict_graph.keys())
    for k, v in dict_graph.items():
        graph.add_edges_from(([(k, t) for t in v]))

    anomalies_start_time = dict()
    for node in graph.nodes:
        # if node != "a":
        short_path = nx.shortest_path(graph, source="a", target=node, weight=None, method='dijkstra')
        anomalies_start_time[node] = anomaly_start + len(short_path) - 1

    if nb_anomalous_data > 100:
        eps = 100
    else:
        eps = 50
    generator = GenerateData(graph)
    data = generator.generate_data(n=nb_data, intervention=True, rootStartIntervention=anomaly_start,
                                   rootEndIntervention=anomaly_start + nb_anomalous_data + eps,
                                   secondInterventionNode=intervention,
                                   seccondStartIntervention=anomalies_start_time[intervention],
                                   secondEndIntervention=anomalies_start_time[
                                                             intervention] + nb_anomalous_data + eps)
    for node in graph.nodes:
        graph.add_edge(node, node)


    rc = micro_cause2(data, data.columns, anomalies_start_time, nb_anomalous_data, gamma_max=3, sig_threshold=0.05)
    print(rc)
