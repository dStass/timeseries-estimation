import numpy as np
from scipy.optimize import leastsq
import pylab as plt

from process_csv.csvrw import CSVReadWrite
from funcs.functions import *

CULL_AMOUNT = 1

M = 0
N = 2

# changes to interactions
special_interactions = {
  'a_0' : 1,
  'c_0' : 0
}

fixed_values = {
  'a_0' : 0.00003,
  'b_0' : 0.055,
  'c_0' : 0.00018,
  'd_0' : -0.048,
}

interactions = []
position_map = {}  # maps a variable name to its position in x and g (for its partial differentiation)

total = 0
# fill gradient, interactions and position_maps
# add polynomial coefficients
for m in range(0, M):
  p_m_name = 'p_' + str(m)
  interactions.append(0)
  position_map[p_m_name] = total
  total += 1

# add sinusoidal coefficients
for char in ['a','b','c','d']:
  for n in range(0, N):

    # add gradient partials
    interaction_amount = 1
    if (char in ['a', 'c']): interaction_amount = 0

    # add to interactions
    interactions.append(interaction_amount)
    
    # add to weights mapping
    sino_n_name = char + '_' + str(n)
    position_map[sino_n_name] = total
    total += 1


for interaction in special_interactions:
  interactions[position_map[interaction]] = special_interactions[interaction]







# read data
training_path = 'detrended-train-data.csv'
csvrw = CSVReadWrite()
training_data = csvrw.csv_to_list(training_path)
col_names = training_data[1]
training_data = [(float(tup[0]), float(tup[1])) for tup in training_data[1:-CULL_AMOUNT]]
initial_guess = [0] * (M+4*N)
for k in fixed_values: initial_guess[position_map[k]] = fixed_values[k]


x = np.array([tup[0] for tup in training_data])
y = np.array([tup[1] for tup in training_data])

# optimize_func = lambda dat: dat[0] + dat[1]*x *np.sin(x[1]*t+x[2]) + x[3] - data
optimize_func = lambda attr: pow(x * attr[0] * np.cos(attr[1]*x) + attr[2]*np.sin(attr[3]*x) - y, 2)

sol = leastsq(optimize_func, initial_guess)[0]
# sol = [float_to_str(s) for s in sol]
print(build_equation(sol, position_map, interactions, M, N), get_loss(x.tolist(), y.tolist(), sol, position_map, interactions, M, N))
print()