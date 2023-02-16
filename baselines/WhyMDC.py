from dowhy.gcm.distribution_change import distribution_change
from dowhy.gcm.cms import ProbabilisticCausalModel
from dowhy.gcm.fcms import AdditiveNoiseModel
from dowhy.gcm.graph import StochasticModel, ConditionalStochasticModel, is_root_node
from dowhy.gcm.stochastic_models import EmpiricalDistribution

from dowhy.gcm.ml.regression import create_linear_regressor, create_lasso_regressor
from dowhy.gcm.divergence import estimate_kl_divergence_continuous

from dowhy.gcm.shapley import ShapleyApproximationMethods


class ShapleyConfig:
    def __init__(self,
                 approximation_method: ShapleyApproximationMethods = ShapleyApproximationMethods.EARLY_STOPPING,
                 num_samples= 100,
                 min_percentage_change_threshold= 0.01,
                 n_jobs= None):
        self.approximation_method = approximation_method
        self.num_samples = num_samples
        self.min_percentage_change_threshold = min_percentage_change_threshold
        self.n_jobs = 2


def whymdc(window_graph, data, anomalous_nodes, targets, anomalies_start_time=None, anomaly_length=200, sig_threshold=0.05):
    rc_list = []

    print(window_graph.edges)
    print(data)
    print(anomalous_nodes)
    print(targets)
    print(anomalies_start_time)

    last_start_time_anomaly = 0
    first_start_time_anomaly = 0
    for node in anomalous_nodes:
        last_start_time_anomaly = max(last_start_time_anomaly, anomalies_start_time[node])
        first_start_time_anomaly = min(last_start_time_anomaly, anomalies_start_time[node])

    first_end_time_anomaly = last_start_time_anomaly + anomaly_length

    anomalous_data = data.loc[last_start_time_anomaly:first_end_time_anomaly]
    normal_data = data.loc[first_start_time_anomaly-anomaly_length-20:first_start_time_anomaly-10]
    # normal_data = data.loc[:first_start_time_anomaly-10]

    model = ProbabilisticCausalModel(window_graph)

    for node in window_graph.nodes:
        if is_root_node(model.graph, node):
            model.set_causal_mechanism(node, EmpiricalDistribution())
        else:
            prediction_model = create_linear_regressor()
            model.set_causal_mechanism(node, AdditiveNoiseModel(prediction_model))

    for node in targets:
        res = distribution_change(causal_model=model, old_data=normal_data, new_data=anomalous_data, target_node=node,
                                  num_samples=anomaly_length, mechanism_change_test_significance_level=sig_threshold,
                                  return_additional_info=True,
                                  difference_estimation_func=estimate_kl_divergence_continuous, mechanism_change_test_fdr_control_method="fdr_tsbh")

        # print(node, res)
        # v = 0
        # rc = None
        # for k in res.keys():
        #     if (abs(res[k]) > v) and (k != node+"-1"):
        #         v = abs(res[k])
        #         rc = k
        # if rc is not None:
        #     if (abs(res[rc]) > 0.05) and (rc not in rc_list):
        #         rc_list.append(rc)

        # did_mecanism_change = res[1]
        # print(node, res[1], res[0])
        # for k in did_mecanism_change.keys():
        #     if (did_mecanism_change[k]) and ((k, node) in window_graph.edges):
        #         rc_list.append(node)
        #         break
        #     if (did_mecanism_change[k]) and (k==node):
        #         rc_list.append(node)

        did_mecanism_change = res[1]
        print(node, res[1])
        for k in did_mecanism_change.keys():
            if (did_mecanism_change[k]) and (k==node):
                rc_list.append(node)
                break

    return rc_list


if __name__ == '__main__':
    import networkx as nx
    import numpy as np
    import pandas as pd

    g = nx.DiGraph()
    g.add_edges_from([("z", "y"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("d", "f"), ("e", "f")])
    anomalous = ["a", "b", "e", "f", "z", "y", "d"]
    anomalies_start = dict()
    for ano in anomalous:
        anomalies_start[ano] = 3500
    anomaly_length = 500

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

    wgraph = nx.DiGraph()
    wgraph.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("e", "f"), ("d", "f"), ("y", "z")])

    res = whymdc(wgraph, data, anomalous, anomalous, anomalies_start_time=anomalies_start, anomaly_length=anomaly_length)
    print(res)
