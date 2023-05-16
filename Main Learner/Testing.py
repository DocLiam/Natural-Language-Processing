from audioop import rms
from DeepLearningOptimized import Data_DL
from DeepLearningOptimized import Model_DL
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy

data_name = input("Data name: ")
model_name = input("Model name: ")

Model = Model_DL.model()
Model.load(model_name)

Data = Data_DL.data()
Data.extract(data_name + "TEST")

model = Word2Vec.load("./DATA/" + data_name[6:] + "RAW.model")

output_values = []

Data.load(Data.input_values[-Model.input_count:], [], stream=Data.stream, shift_count=Data.shift_count)
print(Data.input_values)
Model.test(Data)

text = ""

for i in range(32):
    vector = [float(value) for value in Model.output_values[:Data.shift_count]]
    
    word = model.wv.most_similar(numpy.array(vector))[0][0]
    print(model.wv.most_similar(numpy.array(vector)))
    text += word + " "
    
    Data.load(Data.input_values[Data.shift_count:]+model.wv[word].tolist(), [], stream=Data.stream, shift_count=Data.shift_count)
    
    Model.test(Data)

print(text + "\n")