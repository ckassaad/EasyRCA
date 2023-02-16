# Code Python EasyRCA
EasyRCA is root cause analysis python package that allows to efficiently estimate the root causes of collective anomalies given an acyclic summary causal graph and observational time series from the normal regime as well as time series from the anomalous regime. For more details check our paper: 
C. K. Assaad, I. Ez-zejjari, and L. Zan. Root Cause Identification for Collective Anomalies in Time Series given an Acyclic Summary Causal Graph with Loops. he 26th International Conference on Artificial Intelligence and Statistics. 2023.



## Required python packages

Main packages: networkx, numpy, pandas, itertools, scipy, sklearn, dowhy, causal-learn, tigramite

To install the packages you can use the provided requirement file: requirements.txt

## Test

To test algorithms on simulated data with structural interventions run:
python3 TestEasyRCA/test_simulated_data_with_structural_intervention.py method nb_anomalous_data gamma_max

To test algorithms on simulated data with parametric interventions run:
python3 TestEasyRCA/test_simulated_data_with_parametric_intervention.py method nb_anomalous_data gamma_max

To test algorithms on real data run:
python3 TestEasyRCA/test_real_data.py method gamma_max


method: choose from ["EasyRCA", "EasyRCA*", "MicroCause", "CloudRanger", "WhyMDC"]
nb_anomalous_data: number of timestamps of anomalous data
gamma_max: lag max between a cause and an effect

Example for simulated data: python3 TestEasyRCA/test_simulated_data_with_structural_intervention.py "EasyRCA" 1000 3
Example for real data: python3 TestEasyRCA/test_simulated_data_with_structural_intervention.py "EasyRCA" 3

