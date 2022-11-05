from audioop import rms
from DeepLearner import *
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy

data_name = input("Data name: ")
model_name = input("Model name: ")

Model = Model_Class()
Model.load(model_name)

Data = Data_Class()
Data.extract(data_name + "TEST")

model = Word2Vec.load("./DATA/" + data_name[6:] + "RAW.model")

output_values = []

text = ""

for i in range(100):
    Data.load(Data.input_values[-Model.input_count+len(output_values):]+output_values+[0,0,0,0,0,0,0,0], [], 1, 8)
    Model.test(Data)
    
    vector = numpy.array([float(i) for i in Model.output_values])
    
    word = model.wv.most_similar(vector)[0][0]
    text += word
    
    output_values = Model.output_values.copy()
    
    Data.input_values = Data.input_values[:-Model.output_count]
    
print(text)
