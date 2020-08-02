import math
import random
import numpy as np

from process_csv.csvrw import CSVReadWrite
from funcs.functions import *

# # # # # # # # #
#   VARIABLES   #
# # # # # # # # #

# M (M+1 elements): no. polynomial coefficients, form: p_o + p_1*x + p_2*x^2 + ... + p_M
# N (N+1 elements): no. sinusoidal coefficients of form SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
# M = 2
# N = 2

M = 0
N = 1

BENCHMARK_LOSS = 4.5


# changes to interactions
special_interactions = {
  'a_0' : 1,
  'c_0' : 1
}


fixed_values = {

  # seasonality geogebra
  # 'a_0' : 0.00007625434,
  # 'b_0' : 0.05,
  # 'c_0' : 0.05,
  # 'd_0' : -0.00005775635,

  # seasonality gradient descent
  # loss=0.7320964608205832eqn= 0.00007907479197745075 * x * cos(0.05051324956508964 * x) + -0.00004163450149742875 * x * sin(0.05051749589836491 * x)
  'a_0' : 0.00007907479197745075,
  'b_0' : 0.05051324956508964,
  'c_0' : -0.00004163450149742875,
  'd_0' : 0.05051749589836491,  


  # raw data
  # 'p_0' : 47.22199990800441,
  # 'p_1' : 0.00025142338955179297,
  # 'a_0' : 0.0000180254563817072 ,
  # 'b_0' : 0.050454385282386875,
  # 'c_0' : -0.000037216854880379544,
  # 'd_0' : 0.05007175384854332,
  # 'a_1' : -0.00045322083352745545,
  # 'b_1' : 0.9994464599181256,
  # 'c_1' : -0.03233436835087745,
  # 'd_1' : 0.050941815649287896,


}

CULL_AMOUNT = 1
INTERVAL_SPLIT = 817 - CULL_AMOUNT # 128  767
INTERVAL_STEP = 1  # 64
MODELS_TO_COLLECT = 2000
BINARY_GRANULARITY = 32

# BENCHMARK_LOSS = 1

# assume data has been split
# paths
training_path = 'moving_average_data.csv'
# training_path = 'training.csv'
save_folder = 'output_csvs/'
save_name = 'new_fit'


# # # # # # # # #
#     START     #
# # # # # # # # #

print("Starting app..")


# # # # # # # # #
# PREPROCESSING #
# # # # # # # # #

# read training data
csvrw = CSVReadWrite()
training_data = csvrw.csv_to_list(training_path)
col_names = training_data[1]
training_data = [(float(tup[0]), float(tup[1])) for tup in training_data[1:-CULL_AMOUNT]]

# separate data into x and y lists
x = [tup[0] for tup in training_data]
y = [tup[1] for tup in training_data]

# transform y
# y = [s - (47.222 + 0.0003*t) for s, t in zip(y, x)]

data = [(x[i], y[i]) for i in range(len(x))]

# polynomial + sinusoidal function estimator with gradient descent
# function: SUM(m=0..M){ p_m * x^m } + SUM(n=0..N){ a_n*cos(b_n*x) + c_n*cos(d_n*x) }
# estimates, total size = ((M+1) + (N+1)*4)
# gradient vector:
# first M+1 elements are assigned to p_m in increasing order
# next N+1 elements are assigned to a_n in increasing order
# next N+1 elements are assigned to b_n in increasing order
# next N+1 elements are assigned to c_n in increasing order
# next N+1 elements are assigned to d_n in increasing order
gradient = []  # gradient vector
interactions = [] # keeps a list of x^{interaction} for S1, there should be 2*N interactions (each for cosine and sine parts)
position_map = {}  # maps a variable name to its position in x and g (for its partial differentiation)

total = 0
# fill gradient, interactions and position_maps
# add polynomial coefficients
for m in range(0, M):
  p_m_name = 'p_' + str(m)
  gradient.append(partial('p', m, position_map, interactions))
  interactions.append(0)
  position_map[p_m_name] = total
  total += 1

# add sinusoidal coefficients
for char in ['a','b','c','d']:
  for n in range(0, N):

    # add gradient partials
    interaction_amount = 1
    if (char in ['a', 'c']): interaction_amount = 0

    gradient.append(partial(char, n, position_map, interactions) )  # i.e. add methods aj, bj, cj, dj

    # add to interactions
    interactions.append(interaction_amount)
    
    # add to weights mapping
    sino_n_name = char + '_' + str(n)
    position_map[sino_n_name] = total
    total += 1


for interaction in special_interactions:
  interactions[position_map[interaction]] = special_interactions[interaction]

# # # # # # # # #
#    LEARNING   #
# # # # # # # # #

EPS = 0.00000000005
ALTERNATE_RATE = 0.000005
RATE = 1.0
THRESHOLD_SQ_ERR = 0.01

# INTERVAL_SPLIT = len(training_data) - 1
# INTERVAL_STEP = 1

THRESHOLD_LOSS = (INTERVAL_SPLIT/ 100) * 1
step_count = 0
total = 0

models = {}

# running_norm = 0
# min_weights = None
# min_sqerr = float('inf')
# prev_running_norm = float('inf')
prev_running_norm = float('inf')
prev_weights = None

print("Loss of intercept-gradient model")
weights = [0] * len(gradient)
weights[0] = 47.222
weights[1] = 0.0003
# BENCHMARK_LOSS = get_loss(x, y, weights, position_map, interactions, M, N)

# weights = generate_weights(data, gradient, position_map, fixed_values, interactions, M, N)

print('loss=', BENCHMARK_LOSS)


# BENCHMARK_LOSS = 2.1


fixed_weights = generate_weights(data, gradient, position_map, fixed_values, interactions, M, N)
print("eqn = ", build_equation(fixed_weights, position_map, interactions, M, N))
print("fixed loss: ", get_loss(x, y, fixed_weights, position_map, interactions, M, N))
print("fixed r2=",  get_r_squared(x, y, fixed_weights, position_map, interactions, M, N))
print("sqRes =",  get_squared_residual(x, y, fixed_weights, position_map, interactions, M, N))

# interactions = [1]*len(interactions)
# fixed_weights = generate_weights(data, gradient, position_map, fixed_values, interactions, M, N)
# print("fixed loss2: ", get_loss(x, y, fixed_weights, position_map, interactions, M, N))



print("Begin descent")


interval = 0
while interval + INTERVAL_SPLIT < len(x):
  print("interval: ", interval)
  models[interval] = []

  # gather subset of data
  x_subset = x[interval : interval + INTERVAL_SPLIT]
  y_subset = y[interval : interval + INTERVAL_SPLIT]
  data_subset = [(x_subset[i], y_subset[i]) for i in range(len(x_subset))]

  # initialise weights vector
  weights = generate_weights(data, gradient, position_map, fixed_values, interactions, M, N)

  prev_running_norm = float('inf')
  running_norm = 0
  min_loss = float('inf')

  its = 0
  # rate = 0.00005
  while True:
    reset = False

    its += 1

    # TODO: find a good step length
    
    S = calculate_S(x_subset, weights, position_map, interactions, N, M)
    descent_direction = neg_vector(get_gradient_at(gradient, weights, data_subset, S, N, M))
    # descent_direction[0] = 0
    # descent_direction[1] = 0
    descent_direction = normalise_vector(descent_direction)

    new_rate = 0
    # prev_loss = apply_function(x_subset, weights, position_map, M, N)
    prev_loss = get_loss(x_subset, y_subset, weights, position_map, interactions, M, N, S)
    best_rate = 0
    for i in np.arange(0.000001, 0.001, 0.000001):
      new_rate = i
      new_weights = add_vectors(weights, mul_scalar_to_vec(new_rate, descent_direction))
      curr_loss = get_loss(x_subset, y_subset, new_weights, position_map, interactions, M, N)
      if curr_loss < prev_loss:
        prev_loss = curr_loss
        best_rate = new_rate

    # bin_rate = 1
    # new_rate = bin_rate
    # gran_count = 0
    # while gran_count < BINARY_GRANULARITY:
    #   gran_count += 1
    #   bin_rate /= 2
    #   new_rate = best_rate + bin_rate

    #   new_weights = add_vectors(weights, mul_scalar_to_vec(new_rate, descent_direction))
    #   curr_loss = get_loss(x_subset, y_subset, new_weights, position_map, interactions, M, N)
    #   if curr_loss < prev_loss:
    #     prev_loss = curr_loss
    #     best_rate = new_rate

    rate = best_rate
    if rate == 0:
      rate = ALTERNATE_RATE
      reset = True

    # apply weights
    weights = add_vectors(weights, mul_scalar_to_vec(rate, descent_direction))
    norm = norm_euclidean(get_gradient_at(gradient, weights, data_subset, S, N, M))
    # weights = add_vectors(weights, mul_scalar_to_vec(rate, descent_direction))
    # norm = norm_euclidean(get_gradient_at(gradient, weights, position_map, data_subset, S, N, M))
    running_norm += norm

    # f_applied_at_x = apply_function(x_subset, weights, position_map, M, N)
    # sq_error = get_square_err(f_applied_at_x, y_subset)
    loss_val = get_loss(x_subset, y_subset, weights, position_map, interactions, M, N)
    if loss_val < min_loss:
      min_loss = loss_val
      print("min_loss = ", loss_val)
    if loss_val < BENCHMARK_LOSS:
      eqn = build_equation(weights, position_map, interactions, M, N)
      if loss_val == min_loss: indent = '> '
      else: indent = ''
      print(indent, "loss=", loss_val, "eqn= ", eqn, sep='')
      models[interval].append([loss_val] + [w for w in weights])

      # best model so far
      if loss_val == min_loss:
        fixed_values = {k : weights[position_map[k]] for k in fixed_values}

      # print(fixed_values)

      # weights = generate_weights(data_subset, gradient, position_map, fixed_values, interactions, M, N)
      # prev_running_norm = float('inf')
      # running_norm = 0
      if len(models[interval]) >= MODELS_TO_COLLECT: break
      
    if (its > 1 and its % 50 == 1) or reset:
      if running_norm < prev_running_norm*0.995:
        prev_running_norm = running_norm
        running_norm = 0
      else:
        weights = generate_weights(data_subset, gradient, position_map, fixed_values, interactions, M, N)
        prev_running_norm = float('inf')
        running_norm = 0

  csvrw.list_to_csv(models[interval], save_folder+save_name + '_' + str(interval) + '.csv')
  interval += INTERVAL_STEP


# Saving models
print("Saving models to csv")
interval = 0
col_name = ['interval', 'loss'] \
         + ['p'+str(i) for i in range(M)] \
         + ['a'+str(i) for i in range(N)] \
         + ['b'+str(i) for i in range(N)] \
         + ['c'+str(i) for i in range(N)] \
         + ['d'+str(i) for i in range(N)]

write_out = [col_name]
while interval + INTERVAL_SPLIT < len(x):
  interval_models = models[interval]
  for model in interval_models:
    model_with_interval = [interval] + model
    write_out.append(model_with_interval)
  interval += INTERVAL_STEP

save_name_full = save_name + '_' + str(INTERVAL_SPLIT) + '.csv'
csvrw.list_to_csv(write_out, save_name_full)

print("Task Completed, saved to: ", save_name_full)