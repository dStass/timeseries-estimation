from process_csv.csvrw import CSVReadWrite

# paths
data_path = 'detrended-train-data.csv'
out_path = 'moving_average_data.csv'

# declare csv reader
csvrw = CSVReadWrite()

# read data
data = csvrw.csv_to_list(data_path, shuffle=False, header=False)
cols = data[0]
data = data[1:]
data = [[float(tup[0]), float(tup[1])] for tup in data] 

DATA_POINTS = len(data)

# neighbours
num_nighbours = 20

x = []
y = []
for i in range(DATA_POINTS):
  tup = data[i]
  x.append(tup[0])
  
  neighbours = data[max(0, i-num_nighbours): min(DATA_POINTS, i+num_nighbours+1)]
  y_neighbours = [n[1] for n in neighbours]

  y_avg = sum(y_neighbours) / len(y_neighbours)
  y.append(y_avg)

new_data = [cols]+ [(x[i], y[i]) for i in range(DATA_POINTS)]
csvrw.list_to_csv(new_data, out_path)





