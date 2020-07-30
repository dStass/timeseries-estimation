import math
import random

def build_equation(weights, M, N):
  def build_poly(weight, i):
    weightstr = str(weight)
    if i == 0:
      return weightstr
    elif i == 1:
      return weightstr + ' * x'
    else:
      return weightstr + ' * x^' + str(i)

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

def get_square_err(x, y):
  err = 0
  for i in range(len(x)):
    err += pow((x[i] - y[i]), 2)
  return err/len(x)

# execute function
def apply_function(vec, weights, weights_mapping, M, N):
  to_return = []
  for v in vec:
    to_return.append(f(v, weights, weights_mapping, M, N))
  return to_return


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


# # # # # # # # # # # #
#  function specific  #
# # # # # # # # # # # #

def f(x, weights, weights_mapping, M, N):
  function_val = 0

  # sum polynomials
  for i in range((M+1)):
    function_val += weights[i] * pow(x, i)
  
  # sum sinusoidal
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

def get_loss(x, y, weights, weights_mapping, M, N):
  loss_val = sum([pow(f(x[i], weights, weights_mapping, M, N) - y[i], 2) for i in range(len(x))])
  return loss_val

# # # # # # # # # # # #
#       PARTIALS      #
# # # # # # # # # # # #

# POLYNOMIAL PARTIALS
def pj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    p_j = weights[weights_mapping['p_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += 2 * (pow(x_k, j) * (S1_k + S2_k - y_k))
  return ongoing_sum

# SINUSOIDAL PARTIALS
def aj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    b_j = weights[weights_mapping['b_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += 2 * (math.cos(b_j*x_k) * (S1_k + S2_k - y_k))
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
    ongoing_sum += -2 * (x_k * a_j * math.sin(b_j * x_k) * (S1_k + S2_k - y_k))
  return ongoing_sum

def cj(j, data, weights, weights_mapping, S):
  ongoing_sum = 0
  for tup in data:
    x_k = tup[0]
    y_k = tup[1]
    d_j = weights[weights_mapping['d_'+str(j)]]
    S1_k = S['S1_'+str(x_k)]
    S2_k = S['S2_'+str(x_k)]
    ongoing_sum += 2 * (math.sin(d_j * x_k) * (S1_k + S2_k - y_k))
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
    ongoing_sum += 2 * (x_k * c_j * math.cos(d_j * x_k) * (S1_k + S2_k - y_k))
  return ongoing_sum

# calculate sums S1 and S2 evaluated at data
def calculateS1_at_xk(weights, weights_mapping, N, x_k):
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

def calculateS2_at_xk(weights, weights_mapping, M, x_k):
  ongoing_sum = 0
  for m in range(M):
    # declare coefficients
    p_m = weights[weights_mapping['p_' + str(m)]]

    # build the sum
    ongoing_sum += p_m * pow(x_k, m)
  return ongoing_sum

def calculate_S(x, weights, weights_mapping, N, M):
  S = {}
  for x_k in x:
    key1 = 'S1_' + str(x_k)
    S1_k = calculateS1_at_xk(weights, weights_mapping, N, x_k)
    S[key1] = S1_k

    key2 = 'S2_' + str(x_k)
    S2_k = calculateS2_at_xk(weights, weights_mapping, M, x_k)
    S[key2] = S2_k

  return S


def generate_weights(data, gradient, variable_name_to_position, M, N):
  ACCEPTABLE_LOSS = len(data) / 50
  MAX_SQERR = 0.2
  MIN_NORM = 2000
  loss = ACCEPTABLE_LOSS + 1
  sqerror = MAX_SQERR + 1
  norm = MIN_NORM + 1

  x = []
  y = []
  for tup in data:
    x.append(tup[0])
    y.append(tup[1])

  # weights = None
  # weights = [random.uniform(-0.5,0.5) for _ in range(((M+1) + 4*(N+1)))]
  # generate a close enough point to start
  while sqerror > MAX_SQERR or norm < MIN_NORM:
    weights = [random.uniform(-0.1,0.1) for _ in range(((M+1) + 4*(N+1)))]
    weights[0] = 0
    # weights[1] = 0
    # weights[0] = 47.22
    # weights[1] = 0.0003
    # weights[2] = 0
    fvec = apply_function(x, weights, variable_name_to_position, M, N)
    sqerror = get_square_err(fvec, y)
    sqerror = math.sqrt(sqerror)

    loss = get_loss(x, y, weights, variable_name_to_position, M, N)

    S = calculate_S(x, weights, variable_name_to_position, N, M)
    norm = norm_euclidean(get_gradient_at(gradient, weights, variable_name_to_position, data, S, N, M))
  
  return weights