from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

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
    
    input_count = 32
    output_count = 32
    
    for line in list(model.wv.key_to_index.keys()):
        input_values += model.wv[line].tolist()
        
    input_values = [str(i) for i in input_values]
    
    halfway_point = len(input_values)//2-len(input_values)//2%vector_size
    
    to_write_train = ",".join(input_values[:halfway_point])
    to_write_validate = ",".join(input_values[halfway_point:])
    to_write_test = to_write_train+","+to_write_validate
    
    filew_train = open("STREAM" + name + "TRAIN.txt", "w")
    filew_validate = open("STREAM" + name + "VALIDATE.txt", "w")
    filew_test = open("STREAM" + name + "TEST.txt", "w")
    
    filew_train.write(str(vector_size)+"\n"+to_write_train)
    filew_validate.write(str(vector_size)+"\n"+to_write_validate)
    filew_test.write(str(vector_size)+"\n"+to_write_test)
    
    filew_train.close()
    filew_validate.close()
    filew_test.close()