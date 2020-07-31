import math
import random
import decimal
def build_equation(weights, position_map, interactions, M, N):
  def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

  def build_poly(weight, i):
    weightstr = str(weight)
    if i == 0:
      return weightstr
    else:
      weightstr += ' * x'
      if weight == 1: weightstr == 'x'
      if i == 1:
        return weightstr
      else:
        return weightstr + '^' + str(i)

  # create a new context for this task
  ctx = decimal.Context()

  # 20 digits should be enough for everyone :D
  ctx.prec = 20

  eqn = ''

  for i in range(0, M):
    eqn += build_poly(weights[i], i)
    eqn += ' + '

  for i in range(0, N):
    a_i_pos = (M) + i
    b_i_pos = a_i_pos + (N)
    c_i_pos = b_i_pos + (N)
    d_i_pos = c_i_pos + (N)
    a_i = float_to_str(weights[a_i_pos])
    b_i = float_to_str(weights[b_i_pos])
    c_i = float_to_str(weights[c_i_pos])
    d_i = float_to_str(weights[d_i_pos])
    a_interaction = interactions[a_i_pos]    
    b_interaction = interactions[b_i_pos]
    c_interaction = interactions[c_i_pos]
    d_interaction = interactions[d_i_pos]    
    eqn += '{} * {} * cos({} * {}) + {} * {} * sin({} * {})'.format(a_i,
                                                                    build_poly(1, a_interaction),
                                                                    b_i,
                                                                    build_poly(1, b_interaction),
                                                                    c_i,
                                                                    build_poly(1, c_interaction),
                                                                    d_i,
                                                                    build_poly(1, d_interaction))
    eqn += ' + '
  
  eqn = eqn[:-3]
  return eqn

def get_square_err(x, y):
  err = 0
  for i in range(len(x)):
    err += pow((x[i] - y[i]), 2)
  return err/len(x)

# execute function
def apply_function(vec, weights, position_map, M, N):
  to_return = []
  for v in vec:
    to_return.append(f(v, weights, position_map, M, N))
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

def norm_euclidean(weights):
  ongoing = 0
  for val in weights:
    ongoing += pow(val, 2)
  return math.sqrt(ongoing)


# # # # # # # # # # # #
#  function specific  #
# # # # # # # # # # # #

# loss function
def get_loss(x, y, weights, position_map, interactions, M, N, S = None):
  # loss = 0
  # for i in range(len(x)):
  #   squared_error = 
  loss_val = sum([pow( f(x[i], weights, position_map, interactions, M, N, S) - y[i], 2) for i in range(len(x)) ] )
  return loss_val

# def f_no_interactions(x, weights, position_map, M, N):
#   function_val = 0

#   # sum polynomials
#   for i in range((M+1)):
#     function_val += weights[i] * pow(x, i)
  
#   # sum sinusoidal
#   for i in range(N+1):
#     a_i_pos = (M+1) + i
#     b_i_pos = a_i_pos + (N+1)
#     c_i_pos = b_i_pos + (N+1)
#     d_i_pos = c_i_pos + (N+1)
#     a_i = weights[a_i_pos]
#     b_i = weights[b_i_pos]
#     c_i = weights[c_i_pos]
#     d_i = weights[d_i_pos]
#     function_val += a_i * math.cos(b_i * x) + c_i * math.sin(d_i * x)

#   return function_val

def f(x_k, weights, position_map, interactions, M, N, S = None):
  sum_S1_S2 = 0
  if not S:
    sum_S1_S2 = calculateS1_at_xk(x_k, weights, position_map, interactions, N) \
              + calculateS2_at_xk(x_k, weights, position_map, M)
  
  else:
    sum_S1_S2 = S['S1_' + str(x_k)] + S['S2_' + str(x_k)]
  return sum_S1_S2

# def subfunction(type_name, j, position_map, interaction = 0):

#   def function_pj(j, position_map, interaction):
#     def apply_function_at(x_k, weights):

#     return apply_function_at


#   function_map = {
#     'p' : function_pj,
#     'a' : function_aj,
#     'b' : function_bj,
#     'c' : function_cj,
#     'd' : function_dj
#   }

#   return function_map[type_name](j, position_map, interaction)



# # # # # # # # # # # #
#       PARTIALS      #
# # # # # # # # # # # #

def get_gradient_at(gradient, weights, data, S, N, M):
  evaluated_gradient = []
  total = 0
  for i in range(len(gradient)):
    j = i
    if total >= M:
      j = 0 if N == 0 else (i - M) % N

    g_i_function = gradient[i]
    g_i_eval = g_i_function(data, weights, S)
    evaluated_gradient.append(g_i_eval)
    total += 1
  return evaluated_gradient

# POLYNOMIAL PARTIALS
def partial(type_name, j, position_map, interactions):
  def partial_pj(j, position_map, interactions):
    def apply_partial_at(data, weights, S):
      ongoing_sum = 0
      for tup in data:
        x_k = tup[0]
        y_k = tup[1]
        p_j = weights[position_map['p_'+str(j)]]
        S1_k = S['S1_'+str(x_k)]
        S2_k = S['S2_'+str(x_k)]
        ongoing_sum += 2 * (pow(x_k, j) * (S1_k + S2_k - y_k))
      return ongoing_sum
    return apply_partial_at

# SINUSOIDAL PARTIALS
  def partial_aj(j, position_map, interactions):
    def apply_partial_at(data, weights, S):
      ongoing_sum = 0

      a_pos = position_map['a_'+str(j)]
      a_j = weights[a_pos]
      a_interaction = interactions[a_pos]

      b_pos = position_map['b_'+str(j)]
      b_j = weights[b_pos]
      b_interaction = interactions[b_pos]

      for tup in data:
        x_k = tup[0]
        y_k = tup[1]
        S1_k = S['S1_'+str(x_k)]
        S2_k = S['S2_'+str(x_k)]
        ongoing_sum += 2 * (pow(x_k, a_interaction) * math.cos(b_j * pow(x_k, b_interaction)) * (S1_k + S2_k - y_k))
      return ongoing_sum
    return apply_partial_at

  def partial_bj(j, position_map, interactions):
    def apply_partial_at(data, weights, S):
      ongoing_sum = 0

      a_pos = position_map['a_'+str(j)]
      a_j = weights[a_pos]
      a_interaction = interactions[a_pos]

      b_pos = position_map['b_'+str(j)]
      b_j = weights[b_pos]
      b_interaction = interactions[b_pos]
      for tup in data:
        x_k = tup[0]
        y_k = tup[1]
        S1_k = S['S1_'+str(x_k)]
        S2_k = S['S2_'+str(x_k)]
        ongoing_sum += -2 * (pow(x_k, a_interaction + b_interaction) * a_j * math.sin(b_j * pow(x_k, b_interaction)) * (S1_k + S2_k - y_k))
      return ongoing_sum
    return apply_partial_at

  def partial_cj(j, position_map, interactions):
    def apply_partial_at(data, weights, S):
      c_pos = position_map['c_'+str(j)]
      c_j = weights[c_pos]
      c_interaction = interactions[c_pos]

      d_pos = position_map['d_'+str(j)]
      d_j = weights[d_pos]
      d_interaction = interactions[d_pos]

      ongoing_sum = 0
      for tup in data:
        x_k = tup[0]
        y_k = tup[1]
        S1_k = S['S1_'+str(x_k)]
        S2_k = S['S2_'+str(x_k)]
        ongoing_sum += 2 * (pow(x_k, c_interaction) * math.sin(d_j * pow(x_k, d_interaction)) * (S1_k + S2_k - y_k))
      return ongoing_sum
    return apply_partial_at

  def partial_dj(j, position_map, interactions):
    def apply_partial_at(data, weights, S):

      c_pos = position_map['c_'+str(j)]
      c_j = weights[c_pos]
      c_interaction = interactions[c_pos]

      d_pos = position_map['d_'+str(j)]
      d_j = weights[d_pos]
      d_interaction = interactions[d_pos]

      ongoing_sum = 0
      for tup in data:
        x_k = tup[0]
        y_k = tup[1]
        S1_k = S['S1_'+str(x_k)]
        S2_k = S['S2_'+str(x_k)]
        ongoing_sum += 2 * (pow(x_k, c_interaction + d_interaction) * c_j * math.cos(d_j * pow(x_k, d_interaction)) * (S1_k + S2_k - y_k))
      return ongoing_sum
    return apply_partial_at
  
  # returns the right partial type
  partial_map = {
    'p' : partial_pj,
    'a' : partial_aj,
    'b' : partial_bj,
    'c' : partial_cj,
    'd' : partial_dj
  }
  return partial_map[type_name](j, position_map, interactions)


# calculate sums S1 and S2 evaluated at data
def calculateS1_at_xk(x_k, weights, position_map, interactions, N):
  ongoing_sum = 0
  for n in range(N):
    # declare coefficients
    a_pos = position_map['a_' + str(n)]
    b_pos = position_map['b_' + str(n)]
    c_pos = position_map['c_' + str(n)]
    d_pos = position_map['d_' + str(n)]
    a_n = weights[a_pos]
    b_n = weights[b_pos]
    c_n = weights[c_pos]
    d_n = weights[d_pos]
    a_interaction = interactions[a_pos]
    b_interaction = interactions[b_pos]
    c_interaction = interactions[c_pos]
    d_interaction = interactions[d_pos]

    # build the sum
    ongoing_sum += a_n * pow(x_k, a_interaction) * math.cos(b_n * pow(x_k, b_interaction) * x_k) \
                +  c_n * pow(x_k, c_interaction) * math.sin(d_n * pow(x_k, d_interaction) * x_k)
  return ongoing_sum

def calculateS2_at_xk(x_k, weights, position_map, M):
  ongoing_sum = 0
  for m in range(M):
    # declare coefficients
    p_m = weights[position_map['p_' + str(m)]]

    # build the sum
    ongoing_sum += p_m * pow(x_k, m)
  return ongoing_sum

def calculate_S(x, weights, position_map, interactions, N, M):
  S = {}
  for x_k in x:
    key1 = 'S1_' + str(x_k)
    S1_k = calculateS1_at_xk(x_k, weights, position_map, interactions, N)
    S[key1] = S1_k

    key2 = 'S2_' + str(x_k)
    S2_k = calculateS2_at_xk(x_k, weights, position_map, M)
    S[key2] = S2_k

  return S


def generate_weights(data, gradient, position_map, interactions, M, N):
  ACCEPTABLE_LOSS = len(data) / 10
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
  while loss > ACCEPTABLE_LOSS:
    weights = [random.uniform(-0.1,0.1) for _ in range(((M) + 4*(N)))]
    # weights[0] = 0
    # weights[1] = 0
    weights[0] = 47.222
    weights[1] = 0.0003
    weights[2] = 0
    # fvec = apply_function(x, weights, position_map, M, N)
    # sqerror = get_square_err(fvec, y)
    # sqerror = math.sqrt(sqerror)

    S = calculate_S(x, weights, position_map, interactions, N, M)
    loss = get_loss(x, y, weights, position_map, interactions, M, N, S)

    # norm = norm_euclidean(get_gradient_at(gradient, weights, position_map, data, S, N, M))
  
  return weights