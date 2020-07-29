import math
import random
import numpy as np

from process_csv.csvrw import CSVReadWrite
from funcs.functions import Z, P, A, B, C, D

# # # # # # # # #
#  DEFINITIONS  #
# # # # # # # # #

def build_poly(weight, i):
  weightstr = str(weight)
  if i == 0:
    return weightstr
  elif i == 1:
    return weightstr + ' * x'
  else:
    return weightstr + ' * x^' + str(i)

def build_equation(weights, M, N):
  eqn = ''
  for i in range(0, M+1):
    eqn += build_poly(weights[i], i)
    eqn += ' + '

  for i in range(0, N+1):
    a_i_pos = (M+1) + i
    b_i_pos = a_i_pos + (N+1)
    c_i_pos = b_i_pos + (N+1)
    d_i_pos = c_i_pos + (N+1)
    a_i = weights[a_i_pos]
    b_i = weights[b_i_pos]
    c_i = weights[c_i_pos]
    d_i = weights[d_i_pos]
    eqn += '{} * cos({} * x) + {} * sin({} * x)'.format(a_i, b_i, c_i, d_i)
    eqn += ' + '
  
  eqn = eqn[:-3]
  return eqn


def sqerr(x, y):
  err = 0
  for i in range(len(x)):
    err += pow((x[i] - y[i]), 2)
  return err/len(x)

def f_vec(vec, weights, weights_mapping, M, N):
  to_return = []
  for v in vec:
    to_return.append(f(v, weights, weights_mapping, M, N))
  return to_return

def f(x, weights, weights_mapping, M, N):
  function_val = 0
  for i in range((M+1)):
    function_val += weights[i] * pow(x, i)
  
  for i in range(N+1):
    a_i_pos = (M+1) + i
    b_i_pos = a_i_pos + (N+1)
    c_i_pos = b_i_pos + (N+1)
    d_i_pos = c_i_pos + (N+1)
    a_i = weights[a_i_pos]
    b_i = weights[b_i_pos]
    c_i = weights[c_i_pos]
    d_i = weights[d_i_pos]
    function_val += a_i * math.cos(b_i * x) + c_i * math.sin(d_i * x)
  
  return function_val

def pj(j, data, weights, weights_mapping, S):
  if j == 0: return 0
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    p_j = weights[weights_mapping['p_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    ongoing_sum += 2*j*p_j*(pow(x_k, 2*j-1) + pow(x_k, j-1) * S1_k - y_k * pow(x_k, j-1))
  return ongoing_sum

def aj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    b_j = weights[weights_mapping['b_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += 2*math.cos(b_j*x_k) * (S1_k + S2_k - y_k)
  return ongoing_sum

def bj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    a_j = weights[weights_mapping['a_'+str(j)]]
    b_j = weights[weights_mapping['b_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += -(2 * x_k * a_j * math.sin(b_j * x_k) * (S1_k + S2_k - y_k))
  return ongoing_sum

def cj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    d_j = weights[weights_mapping['d_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += 2 * math.sin(d_j * x_k) * (S1_k + S2_k - y_k)
  return ongoing_sum

def dj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    c_j = weights[weights_mapping['c_'+str(j)]]
    d_j = weights[weights_mapping['d_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += 2 * x_k * c_j * math.cos(d_j * x_k) * (S1_k + S2_k - y_k)
  return ongoing_sum

def calculateS1_at_xk(data, weights, weights_mapping, N, x_k):
  ongoing_sum = 0
  for n in range(N):
    # declare coefficients
    a_n = weights[weights_mapping['a_' + str(n)]]
    b_n = weights[weights_mapping['b_' + str(n)]]
    c_n = weights[weights_mapping['c_' + str(n)]]
    d_n = weights[weights_mapping['d_' + str(n)]]

    # build the sum
    ongoing_sum += a_n * math.cos(b_n * x_k) + c_n * math.sin(d_n * x_k)
  return ongoing_sum

def calculateS2_at_xk(data, weights, weights_mapping, M, x_k):
  ongoing_sum = 0
  for m in range(M):
    # declare coefficients
    p_m = weights[weights_mapping['p_' + str(m)]]

    # build the sum
    ongoing_sum += p_m * pow(x_k, m)
  return ongoing_sum

def calculate_S(data, weights, weights_mapping, N, M):
  S = {}
  for tup in data:
    x_k = tup[0]      
    key = 'S1_' + str(x_k)
    S1_k = calculateS1_at_xk(data, weights, weights_mapping, N, x_k)
    S[key] = S1_k

  for tup in data:
    x_k = tup[0]      
    key = 'S2_' + str(x_k)
    S2_k = calculateS2_at_xk(data, weights, weights_mapping, M, x_k)
    S[key] = S2_k

  return S



# vector methods

def mul_scalar_to_vec(scalar, vec):
  return [scalar * v for v in vec]

def add_vectors(vec0, vec1):
  to_return = []
  for i in range(len(vec0)):
    to_return.append(vec0[i] + vec1[i])
  return to_return

def normalise_vector(vec0):
  vec_norm = norm_euclidean(vec0)
  to_return = []
  for v in vec0:
    to_return.append(v/vec_norm)
  return to_return

def neg_vector(vec):
  return [-v for v in vec]

def get_gradient_at(gradient, weights, weights_mapping, data, S, N, M):
  evaluated_gradient = []
  total = 0
  for i in range(len(gradient)):
    j = i
    if total >= M:
      j = (i - M) % N

    g_i_function = gradient[i]
    g_i_eval = g_i_function(j, data, weights, weights_mapping, S)
    evaluated_gradient.append(g_i_eval)
    total += 1
  return evaluated_gradient

def norm_euclidean(weights):
  ongoing = 0
  for val in weights:
    ongoing += pow(val, 2)
  return math.sqrt(ongoing)


# # # # # # # # #
#     START     #
# # # # # # # # #

print("Starting app..")


# # # # # # # # #
# PREPROCESSING #
# # # # # # # # #

# assume data has been split
# paths
training_path = 'training2.csv'

# read training data
csvrw = CSVReadWrite()
training_data = csvrw.csv_to_list(training_path)
training_data = [(float(tup[0]), float(tup[1])) for tup in training_data]
x = [tup[0] for tup in training_data]
y = [tup[1] for tup in training_data]

# polynomial + sinusoidal function estimator with gradient descent
# set up variables:
variables = {}

# set up memoized
memoized = {}
for k in ['P', 'A', 'B', 'C', 'D']:
  memoized[k] = {}

# M (M+1 elements): no. polynomial coefficients, form: p_o + p_1*x + p_2*x^2 + ... + p_M
# N (N+1 elements): no. sinusoidal coefficients of form SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
M = 1
N = 3

# function: SUM(m=0..M){ p_m * x^m } + SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
# gradient, first M+1 elements are gradients of polynomial terms
gradient = []  # gradient vector
variable_name_to_position = {}  # maps a variable name to its position in x and g (for its partial differentiation)
total = 0


# estimates, total size = ((M+1) + (N+1)*4)
# first M+1 elements are assigned to p_m in increasing order
# next N+1 elements are assigned to a_n in increasing order
# next N+1 elements are assigned to b_n in increasing order
# next N+1 elements are assigned to c_n in increasing order
# next N+1 elements are assigned to d_n in increasing order

# fill variable_name_to_position
# add polynomial coefficients
for m in range(0, M+1):
  p_m_name = 'p_' + str(m)
  variable_name_to_position[p_m_name] = total
  gradient.append(pj)
  total += 1

# add sinusoidal coefficients
for char in ['a','b','c','d']:
  for n in range(0, N+1):
    a_n_name = char + '_' + str(n)
    variable_name_to_position[a_n_name] = total
    gradient.append(globals()[char+'j'])
    total += 1

MIN_SQERR = 0.2
MIN_NORM = 2000
sqerror = MIN_SQERR + 1
norm = MIN_NORM + 1


while sqerror > MIN_SQERR or norm > MIN_NORM:
  weights = [random.uniform(-0.5,0.5) for _ in range(((M+1) + 4*(N+1)))]
  weights[0] = 0
  weights[1] = 0
  # weights[0] = 47.22
  # weights[1] = 0.0003
  # weights[2] = 0
  fvec = f_vec(x, weights, variable_name_to_position, M, N)
  sqerror = sqerr(fvec, y)

  S = calculate_S(training_data, weights, variable_name_to_position, N, M)
  norm = norm_euclidean(get_gradient_at(gradient, weights, variable_name_to_position, training_data, S, N, M))
  print()




# # # # # # # # #
#    LEARNING   #
# # # # # # # # #

EPS = 0.0005
rate = 0.000001
MAX_STEPS = 10000
step_count = 0
total = 0

norms = []
running_norm = 0
min_weights = None
min_sqerr = float('inf')

print("begin")

# while step_count < MAX_STEPS:
while True:
  # if step_count == MAX_STEPS:
  #   weights = [random.uniform(0, 1) for _ in range(((M+1) + 4*(N+1)))]

  S = calculate_S(training_data, weights, variable_name_to_position, N, M)
  step = neg_vector(get_gradient_at(gradient, weights, variable_name_to_position, training_data, S, N, M))
  # step = get_gradient_at(gradient, weights, variable_name_to_position, training_data, S, N, M)
  # step = normalise_vector(step)
  weights = add_vectors(weights, mul_scalar_to_vec(rate, step))
  norm = norm_euclidean(get_gradient_at(gradient, weights, variable_name_to_position, training_data, S, N, M))
  # norms.append((str(step_count), str(norm)))
  running_norm += norm
  fvec = f_vec(x, weights, variable_name_to_position, M, N)
  sqerror = sqerr(fvec, y)
  if total > 0 and total % 50 == 0:
    print("sqerr = ", sqerror)
    print("avgnorm = ", running_norm / total)

  
  if sqerror < min_sqerr:
    # min_norm = norm
    min_sqerr = sqerror
    min_weights = [w for w in weights]
    text = "cycle= " + str(total) + " min_sqerr = " + str(min_sqerr) + ", eqn=" + build_equation(weights, M, N) + '\n\n\n'
    with open("log.txt", "a") as myfile:
      myfile.write(text)
    step_count = 0
  # print("avg=", sum(norms)/len(norms), "min=", min(norms))
  step_count += 1
  total += 1


# save_path = 'save_output.csv'
# csvrw.list_to_csv(norms, save_path)
# print(norms)



# print(get_g_at_x(g, x))
# print(norm_euclidean(get_g_at_x(g, x)))
# while (norm_euclidean(get_gradient_at(gradient, weights)) > eps):
#   if step > MAX_STEPS: break
#   step = neg_vector(get_gradient_at(g, x))
#   x = add_vectors(x, mul_scalar_to_vec(rate, s))
  
#   for k in variable_name_to_position:
#     position = variable_name_to_position[k]
#     variables[k] = x[position]
  
#   step += 1

#   norm = norm_euclidean(get_g_at_x(g, x))


# print("norm:", norm, "x=\n", x)
