import math
import random
import numpy as np

from process_csv.csvrw import CSVReadWrite
from funcs.functions import Z, P, A, B, C, D

# # # # # # # # #
#  DEFINITIONS  #
# # # # # # # # #

def mul_scalar_to_vec(scalar, vec):
  return [scalar * v for v in vec]

def add_vectors(vec0, vec1):
  to_return = []
  for i in range(len(vec0)):
    to_return.append(vec0[0] + vec1[i])
  return to_return

def neg_vector(vec):
  return [-v for v in vec]

def get_g_at_x(gradient, x):
  g_at_x = []
  for i in range(len(gradient)):
    g_i = gradient[i]
    x_i = x[i]
    g_at_x.append(g_i.execute(x_i))
  return g_at_x

def norm_euclidean(x):
  ongoing = 0
  for val in x:
    ongoing += pow(val, 2)
  return math.sqrt(ongoing)


# # # # # # # # #
# PREPROCESSING #
# # # # # # # # #

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
x = [random.uniform(0, 1) for _ in range(((M+1) + 4*(N+1)))]

# function: SUM(m=0..M){ p_m * x^m } + SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
# gradient, first M+1 elements are gradients of polynomial terms
g = []  # gradient vector
variable_name_to_position = {}  # maps a variable name to its position in x and g (for its partial differentiation)
total = 0

# add polynomial partials
for m in range(0, M+1):
  p_m_name = 'p_' + str(m)
  variable_name_to_position[p_m_name] = total
  if m == 0:
    func = Z(variables, memoized)
  else:
    func = P(variables, memoized, p_m_name, m, m-1)
  g.append(func)
  total += 1

# add sinusoidal partials

# a functions:
for n in range(0, N+1):
  a_n_name = 'a_' + str(n)
  b_n_name = 'b_' + str(n)
  variable_name_to_position[a_n_name] = total
  func = A(variables, memoized, a_n_name)
  g.append(func)
  total += 1

# b functions:
for n in range(0, N+1):
  a_n_name = 'a_' + str(n)
  b_n_name = 'b_' + str(n)
  variable_name_to_position[b_n_name] = total
  func = B(variables, memoized, a_n_name, b_n_name)
  g.append(func)
  total += 1

# c functions:
for n in range(0, N+1):
  c_n_name = 'c_' + str(n)
  d_n_name = 'd_' + str(n)
  variable_name_to_position[c_n_name] = total
  func = C(variables, memoized, d_n_name)
  g.append(func)
  total += 1

# d functions:
for n in range(0, N+1):
  c_n_name = 'c_' + str(n)
  d_n_name = 'd_' + str(n)
  variable_name_to_position[d_n_name] = total
  func = D(variables, memoized, c_n_name, d_n_name)
  g.append(func)
  total += 1

# populate variables (must be copied every time ... better way to fix this?)
for k in variable_name_to_position:
  position = variable_name_to_position[k]
  variables[k] = x[position]


# # # # # # # # #
#    LEARNING   #
# # # # # # # # #

eps = 0.05
rate = 0.4
MAX_STEPS = 2000
step = 0

# print(get_g_at_x(g, x))
# print(norm_euclidean(get_g_at_x(g, x)))
while (norm_euclidean(get_g_at_x(g, x)) > eps):
  if step > MAX_STEPS: break
  s = neg_vector(get_g_at_x(g, x))
  x = add_vectors(x, mul_scalar_to_vec(rate, s))
  
  for k in variable_name_to_position:
    position = variable_name_to_position[k]
    variables[k] = x[position]
  
  step += 1

  norm = norm_euclidean(get_g_at_x(g, x))


print("norm:", norm, "x=\n", x)
