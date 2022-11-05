from DeepLearner import *

data_name = input("Data name: ")
model_name = input("Model name: ")

model_count = int(input("Model count: "))

for i in range(model_count):
    Model = Model_Class()
    Model.load(model_name + str(i), min_diff=0.00001, learning_rate=0.0000001, cycles=10, hidden_shaped=False, normaliser_depth=0)

    Data = Data_Class()
    Data.extract(data_name)

    Model.train(Data)
    Model.save()