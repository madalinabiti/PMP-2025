# 1 a
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
from hmmlearn import hmm
import numpy as np

# Definim structura retelei
model = DiscreteBayesianNetwork([('O', 'H'), ('O', 'W'), ('H', 'R'), ('W', 'R'), ('H', 'E'),('R','C')])

# CPDs
cpd_O = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]])
cpd_H = TabularCPD(variable='H', variable_card=2, 
                   values=[[0.1, 0.8],   # H=0 
                           [0.9, 0.2]],  # H=1
                   evidence=['O'], evidence_card=[2])
cpd_W = TabularCPD(variable='W', variable_card=2,
                   values=[[0.9, 0.4],   # W=0
                           [0.1, 0.6]],  # W=1
                   evidence=['O'], evidence_card=[2])
cpd_R = TabularCPD(variable='R', variable_card=2,
                   values=[[0.4, 0.1, 0.7, 0.5],  # R=0
                           [0.6, 0.9, 0.3, 0.5]], # R=1
                   evidence=['H', 'W'], evidence_card=[2, 2])
cpd_E = TabularCPD(variable='E', variable_card=2,
                   values=[[0.2, 0.8],   # E=0
                           [0.8, 0.2]],  # E=1
                   evidence=['H'], evidence_card=[2])
cpd_C = TabularCPD(variable='C', variable_card=2,
                   values=[[0.15, 0.6],   # C=0
                           [0.85, 0.4]],  # C=1
                   evidence=['R'], evidence_card=[2])

# Adaugam CPD-urile in model
model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)

# Verificam consistenta
assert model.check_model()

# Inferenta
infer = VariableElimination(model)
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

# Performing exact inference using Variable Elimination
# 1b
result = infer.query(variables=['H'], evidence={'C': 1})
print(result)



infer = VariableElimination(model)
result = infer.query(variables=['E'], evidence={'C': 1})
print(result)

# infer = VariableElimination(model)
# result = infer.query(variables=['E'], evidence={'C': 1})
# print(result)


# 1 c
# W⊥E|H : W  si E sunt independente una fata de cealalta, dar ambele sunt dependente de E
# O este independet fata de R, dar C este dependent de R 


# 2 a
# Mapping note -> index
obs_map = {'W':0, 'R':1, 'L':2}

# Definim HMM
model = hmm.MultinomialHMM(n_components=3, init_params="")
model.startprob_ = np.array([1/4, 1/3, 1/3])  # probabilitati initiale
model.transmat_ = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.25]
])
model.emissionprob_ = np.array([
    [0.1, 0.05, 0.8],  # L
    [0.7,0.25,0.15],  # M
    [0.2,0.7,0.05]   # H
])

# 2 b
observations = ['M','H', 'L']
obs_seq = np.array([obs_map[o] for o in observations]).reshape(-1,1)

prob = model.score(obs_seq)
print("Log-probabilitatea secventei:", prob)
print("Probabilitatea secventei:", np.exp(prob))