from process_csv.csvrw import CSVReadWrite

# paths
data_path = 'data.csv'
training_path = 'training.csv'
testing_path = 'test.csv'

# amounts
training_percent = 0.8
testing_percent = 1 - training_percent

# declare csv reader
csvrw = CSVReadWrite()

# read data
data = csvrw.csv_to_list(data_path, shuffle=True)

# create training and testing data
num_training = int(len(data) * training_percent)
training_data = sorted(data[:num_training], key=lambda x: int(x[0]))
testing_data = sorted(data[num_training:], key=lambda x: int(x[0]))

# save training and testing data
csvrw.list_to_csv(training_data, training_path)
csvrw.list_to_csv(testing_data, testing_path)
