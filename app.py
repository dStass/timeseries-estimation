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
M = 1
N = 4

# assume data has been split
# paths
training_path = 'training.csv'
save_folder = 'output_csvs/'
save_name = 'fitted_coefficients'


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
training_data = [(float(tup[0]), float(tup[1])) for tup in training_data]

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
variable_name_to_position = {}  # maps a variable name to its position in x and g (for its partial differentiation)
total = 0


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
    gradient.append(globals()[char+'j'])  # i.e. add methods aj, bj, cj, dj
    total += 1

# # # # # # # # #
#    LEARNING   #
# # # # # # # # #

EPS = 0.00000000005
RATE = 1.0
THRESHOLD_SQ_ERR = 0.01
INTERVAL_SPLIT = len(x) - 1 # int(len(x)/2)
THRESHOLD_LOSS = (INTERVAL_SPLIT/ 100) * 0.8
step_count = 0
total = 0
MODELS_TO_COLLECT = 5

models = {}

# running_norm = 0
# min_weights = None
# min_sqerr = float('inf')
# prev_running_norm = float('inf')
prev_running_norm = float('inf')
prev_weights = None

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
  weights = generate_weights(data, gradient, variable_name_to_position, M, N)

  prev_running_norm = float('inf')
  running_norm = 0

  its = 0
  rate = 0.00005
  while True:
    its += 1

    # TODO: find a good step length
    
    S = calculate_S(x_subset, weights, variable_name_to_position, N, M)
    descent_direction = neg_vector(get_gradient_at(gradient, weights, variable_name_to_position, data_subset, S, N, M))
    descent_direction = normalise_vector(descent_direction)
    weights = add_vectors(weights, mul_scalar_to_vec(rate, descent_direction))
    norm = norm_euclidean(get_gradient_at(gradient, weights, variable_name_to_position, data_subset, S, N, M))
    running_norm += norm

    f_applied_at_x = apply_function(x_subset, weights, variable_name_to_position, M, N)
    sq_error = get_square_err(f_applied_at_x, y_subset)
    loss_val = get_loss(x_subset, y_subset, weights, variable_name_to_position, M, N)
    if loss_val < THRESHOLD_LOSS:
      eqn = build_equation(weights, M, N)
      print("loss=", loss_val, "eqn= ", eqn)
      models[interval].append([sq_error] + [w for w in weights])
      weights = generate_weights(data_subset, gradient, variable_name_to_position, M, N)
      prev_running_norm = float('inf')
      running_norm = 0
      if len(models[interval]) >= MODELS_TO_COLLECT: break
      
    if its > 1 and its % 50 == 1:
      if running_norm < prev_running_norm*1.0:
        prev_running_norm = running_norm
        running_norm = 0
      else:
        weights = generate_weights(data, gradient, variable_name_to_position, M, N)
        prev_running_norm = float('inf')
        running_norm = 0
      print(loss_val, prev_running_norm)
  csvrw.list_to_csv(models[interval], save_folder+save_name + '_' + str(interval) + '.csv')
  interval += 1


# Saving models
print("Saving models to csv")
interval = 0
col_name = ['interval', 'sq_error'] \
         + ['p'+str(i) for i in range(M+1)] \
         + ['a'+str(i) for i in range(N+1)] \
         + ['b'+str(i) for i in range(N+1)] \
         + ['c'+str(i) for i in range(N+1)] \
         + ['d'+str(i) for i in range(N+1)]

write_out = [col_name]
while interval + INTERVAL_SPLIT < len(x):
  interval_models = models[interval]
  for model in interval_models:
    model_with_interval = [interval] + model
    write_out.append(model_with_interval)
  interval += 1

csvrw.list_to_csv(write_out, save_name + '_' + str(INTERVAL_SPLIT) + '_' + str(interval) + '.csv')

print("Task Completed")








# interval = 0
# while interval + INTERVAL_SPLIT < len(x):
#   models[interval] = []

#   # gather subset of data
#   x_subset = x[interval : interval + INTERVAL_SPLIT]
#   y_subset = y[interval : interval + INTERVAL_SPLIT]
#   data_subset = [(x_subset[i], y_subset[i]) for i in range(len(x_subset))]

#   # initialise weights vector
#   weights = generate_weights(data, gradient, variable_name_to_position, M, N)

#   prev_running_norm = 0
#   running_norm = 0

#   its = 0
#   sq_err = float('inf')
#   while True:
#     its += 1
#     # get squared sums and generate step/descent direction
#     S = calculate_S(x, weights, variable_name_to_position, N, M)
#     descent_direction = neg_vector(get_gradient_at(gradient, weights, variable_name_to_position, data_subset, S, N, M))
#     descent_direction = normalise_vector(descent_direction)

#     # build step
#     rate = RATE
#     step = 0
#     while True:
#       new_step = step + rate
#       rate /= 2
#       new_weights = add_vectors(weights, mul_scalar_to_vec(new_step, descent_direction))
#       f_x = apply_function(x_subset, new_weights, variable_name_to_position, M, N)

#       # get squared error
#       new_sq_err = get_square_err(f_x, y_subset)
#       if new_sq_err <= sq_err:
#         sq_err = new_sq_err
#         step = new_step
    
#       if rate <= EPS: break
    
#     print(sq_err, rate, norm)
#     if step == 0:
#       weights = generate_weights(data_subset, gradient, variable_name_to_position, M, N)
#       running_norm = 0
#       prev_running_norm = 0
#       sq_err = float('inf')
#       continue
#     weights = add_vectors(weights, mul_scalar_to_vec(step, descent_direction))
#     norm = norm_euclidean(get_gradient_at(gradient, weights, variable_name_to_position, data_subset, S, N, M))
#     running_norm += norm
#     if sq_err < THRESHOLD_SQ_ERR:
#       if running_norm < prev_running_norm:
#         prev_running_norm = running_norm
#         prev_weights = weights
#       else:
#         models[interval].append([w for w in prev_weights])
#         eqn = build_equation(weights, M, N)
#         text = "cycle= " + str(total) + " min_sqerr = " + str(min_sqerr) + ", eqn=" + eqn + '\n\n\n'
#         print("sqerr=", sqerror, ", eqn = ", eqn)
#         if len(models[interval]) > MODELS_TO_COLLECT: break
#     else:
#       if its % 100 == 0 and running_norm > prev_running_norm:
#         weights = generate_weights(data_subset, gradient, variable_name_to_position, M, N)
#         sq_err = float('inf')
#         running_norm = 0
#         prev_running_norm = 0
#         continue

#   interval += 1
  



# while True:

#   S = calculate_S(x, weights, variable_name_to_position, N, M)
#   step = neg_vector(get_gradient_at(gradient, weights, variable_name_to_position, data, S, N, M))
#   step = normalise_vector(step)
#   local_rate = rate
#   while True:
#     # step = get_gradient_at(gradient, weights, variable_name_to_position, training_data, S, N, M)
#     new_weights = add_vectors(weights, mul_scalar_to_vec(local_rate, step))
    
#     # norms.append((str(step_count), str(norm)))
#     fvec = apply_function(x, new_weights, variable_name_to_position, M, N)
#     new_sqerror = sqerr(fvec, y)
#     if new_sqerror < sqerror:
#       weights = new_weights
#       sqerror = new_sqerror
#       break
#     local_rate /= 2
#     if local_rate == 0:
#       weights = weights = generate_weights(data, gradient, variable_name_to_position, M, N)
#       break


#   norm = norm_euclidean(get_gradient_at(gradient, weights, variable_name_to_position, data, S, N, M))
#   running_norm += norm
#   if total > 0 and total % 50 == 0:
#     print("sqerr = ", sqerror)
#     print("avgnorm = ", running_norm / total)
#     if running_norm < prev_running_norm:
#       prev_running_norm = running_norm
#     else:
#       weights = weights = generate_weights(data, gradient, variable_name_to_position, M, N)

  
#   if sqerror < min_sqerr or sqerror < THRESHOLD_SQERR:
#     # min_norm = norm
#     min_sqerr = min(min_sqerr, sqerror)
#     min_weights = [w for w in weights]
#     eqn = build_equation(weights, M, N)
#     text = "cycle= " + str(total) + " min_sqerr = " + str(min_sqerr) + ", eqn=" + eqn + '\n\n\n'
#     print("sqerr=", sqerror, ", eqn = ", eqn)
#     with open("log.txt", "a") as myfile:
#       myfile.write(text)
#     step_count = 0
#   # print("avg=", sum(norms)/len(norms), "min=", min(norms))
#   step_count += 1
#   total += 1