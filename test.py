import os
import pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict.keys())