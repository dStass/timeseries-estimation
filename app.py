import numpy as np

from process_csv.csvrw import CSVReadWrite
from funcs.functions import Z, P, A, B, C, D


# assume data has been split
# paths
training_path = 'training.csv'

# read training data
csvrw = CSVReadWrite()
training_data = csvrw.csv_to_list(training_path)

# polynomial + sinusoidal function estimator with gradient descent
# set up variables:
variables = {}

# set up memoized
memoized = {}
for k in ['P', 'A', 'B', 'C', 'D']:
  memoized[k] = {}

# M (M+1 elements): no. polynomial coefficients, form: p_o + p_1*x + p_2*x^2 + ... + p_M
# N (N+1 elements): no. sinusoidal coefficients of form SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
M, N = 3, 3

# estimates, total size = ((M+1) + (N+1)*4)
# first M+1 elements are assigned to p_m in increasing order
# next N+1 elements are assigned to a_n in increasing order
# next N+1 elements are assigned to b_n in increasing order
# next N+1 elements are assigned to c_n in increasing order
# next N+1 elements are assigned to d_n in increasing order
x = [1] * ((M+1) + 4*(N+1))

# function: SUM(m=0..M){ p_m * x^m } + SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
# gradient, first M+1 elements are gradients of polynomial terms
g = []  # gradient vector
variable_name_to_position = {}  # maps a variable name to its position in x and g (for its partial differentiation)
total = 0

for m in range(0, M+1):
  p_m_name = 'p_' + str(m)
  variable_name_to_position[p_m_name] = total
  if m == 0:
    func = Z(variables, memoized)
  else:
    func = P(variables, memoized, p_m_name, m, m-1)
  g.append(func)
  total += 1

# 
for n in range(0, N+1):


# populate variables
for k in variable_name_to_position:
  position = variable_name_to_position[k]
  variables[k] = x[position]

print()
# for n in range(0, N+1):
