import math
class Function:
  def __init__(self, variables, memoized):
    self.variables = variables
    self.memoized = memoized
    pass

  def execute(self, x):
    pass

class Z(Function):
  """
  returns zero
  """
  def execute(self, x=None):
    return 0
  
class P(Function):
  """
  returns val * p_m * x^p
  """

  def __init__(self, variables, memoized, p_m_name, val, p):
    self.variables = variables
    self.memoized = memoized
    self.p_m_name = p_m_name
    self.val = val
    self.p = p

  def execute(self, x):
    p_m = self.variables[self.p_m_name]
    val = self.val
    p = self.p

    memoized = self.memoized['P']
    key = (p_m, val, p)
    
    if key not in memoized:
      memoized[key] = val * p_m * pow(x,p)
    return memoized[key]


class A(Function):
  """
  returns cos(b_n * x)
  """

  def __init__(self, variables, memoized, b_n_name):
    self.variables = variables
    self.memoized = memoized
    self.b_n_name = b_n_name

  def execute(self, x):
    b_n = self.variables[self.b_n_name]
    memoized = self.memoized['A']
    key = (x,b_n)
    if key not in memoized:
      memoized[key] = math.cos(b_n*x)
    return memoized[key]
  
class B(Function):
  """
  returns - a_n * b_n * sin(b_n * x)
  """

  def __init__(self, variables, memoized, a_n_name, b_n_name):
    self.variables = variables
    self.memoized = memoized
    self.a_n_name = a_n_name
    self.b_n_name = b_n_name

  def execute(self, x):
    a_n = self.variables[self.a_n_name]
    b_n = self.variables[self.b_n_name]
    memoized = self.memoized['B']
    key = (x,a_n,b_n)
    if key not in memoized:
      memoized[key] = - a_n * b_n * math.sin(b_n * x)
    return memoized[key]


class C(Function):
  """
  returns sin(d_n * x)
  """

  def __init__(self, variables, memoized, d_n_name):
    self.variables = variables
    self.memoized = memoized
    self.d_n_name = d_n_name

  def execute(self, x):
    d_n = self.variables[self.d_n_name]
    memoized = self.memoized['C']
    key = (x,d_n)
    if key not in memoized:
      memoized[key] = math.sin(d_n*x)
    return memoized[key]

class D(Function):
  """
  returns: c_n * d_n * cos(d_n * x)
  """

  def __init__(self, variables, memoized, c_n_name, d_n_name):
    self.variables = variables
    self.memoized = memoized
    self.c_n_name = c_n_name
    self.d_n_name = d_n_name

  def execute(self, x):
    c_n = self.variables[self.c_n_name]
    d_n = self.variables[self.d_n_name]
    memoized = self.memoized['D']
    key = (x,c_n,d_n)
    if key not in memoized:
      memoized[key] = c_n * d_n * math.cos(d_n * x)
    return memoized[key]

# memoized = {
#   "P" : {},
#   "A" : {},
#   "B" : {}
# }

# variables = {
#   "a_0" : 1,
#   "b_0" : 1,
# }

# b = B(variables, memoized, 'a_0', 'b_0')
# print(b.execute(2))