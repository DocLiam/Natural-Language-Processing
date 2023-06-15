from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy

name = "SHAKESPEARE"

filer = open(name + "RAW.txt", "r").read().replace("\n", " ")

data = []

for i in sent_tokenize(filer):
    temp = []
    
    for j in word_tokenize(i):
        temp.append(j.lower())
        
    data.append(temp)
    
try:
    model = Word2Vec.load(name + "RAW.model")
except:  
    model = Word2Vec(data, vector_size=8, min_count=1, window=16)
    model.save(name + "RAW.model")
finally:
    print(model.wv.key_to_index.keys())
    print(data)
    input_values = []
    
    vector_size = 8
    
    input_count = 64
    
    for sentence in data:
        for word in sentence:
            input_values += model.wv[word].tolist()
        
    input_values = [str(i) for i in input_values]
    
    to_write_train = ""
    to_write_validate = ""
    to_write_test = ""
    
    for i in range((len(input_values)-input_count)//vector_size):
        if i%3 == 0:
            to_write_validate += ",".join(input_values[i*vector_size:i*vector_size+input_count]) + ":1.0\n"
            to_write_validate += ",".join(input_values[i*vector_size:i*vector_size+input_count-vector_size]) + "," + ",".join([str(k) for k in model.wv[model.wv.most_similar(numpy.array([float(k) for k in input_values[i*vector_size+input_count-vector_size:i*vector_size+input_count]]))[-1][0]].tolist()]) + ":0.5\n"
        else:
            to_write_train += ",".join(input_values[i*vector_size:i*vector_size+input_count]) + ":1.0\n"
            to_write_train += ",".join(input_values[i*vector_size:i*vector_size+input_count-vector_size]) + "," + ",".join([str(k) for k in model.wv[model.wv.most_similar(numpy.array([float(k) for k in input_values[i*vector_size+input_count-vector_size:i*vector_size+input_count]]))[-1][0]].tolist()]) + ":0.5\n"

        to_write_test += ",".join(input_values[i*vector_size:i*vector_size+input_count]) + ":1.0\n"
        to_write_test += ",".join(input_values[i*vector_size:i*vector_size+input_count-vector_size]) + "," + ",".join([str(k) for k in model.wv[model.wv.most_similar(numpy.array([float(k) for k in input_values[i*vector_size+input_count-vector_size:i*vector_size+input_count]]))[-1][0]].tolist()]) + ":0.5\n"
        
    filew_train = open(name + "TRAIN.txt", "w")
    filew_validate = open(name + "VALIDATE.txt", "w")
    filew_test = open(name + "TEST.txt", "w")
    
    filew_train.write(to_write_train[:-1])
    filew_validate.write(to_write_validate[:-1])
    filew_test.write(to_write_test[:-1])
    
    filew_train.close()
    filew_validate.close()
    filew_test.close()