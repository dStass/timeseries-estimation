import csv
import random

data_path = 'data.csv'
training_path = 'training.csv'
test_path = 'test.csv'


training_percent = 0.8
testing_percent = 1 - training_percent

data = []
with open(data_path, newline='') as csvfile:
  csvreader = csv.reader(csvfile, delimiter=',')
  for row in csvreader:
    data.append(row)

random.shuffle(data)

num_training = int(len(data) * training_percent)

training_data = sorted(data[:num_training], key=int)
testing_data = sorted(data[num_training:], key=int)

with open(training_path, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  for row in training_data:
    writer.writerow(row)

with open(test_path, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  for row in testing_data:
    writer.writerow(row)