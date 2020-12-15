import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import json
from tensorflow import keras

model = keras.models.load_model('model_5')
f = open('data/test.jsonl')
ids = []
data = []
for line in f:
    jLine = json.loads(line)
    ids.append(jLine["id"])
    data.append(jLine["response"])
f.close()
output = model.predict(data)
print("output length " + str(len(output)))

f = open("answer.txt", "w")
for i in range(len(ids)):
    result = "SARCASM"
    if output[i] < 0:
        result = "NOT_SARCASM"
    f.write(str(ids[i]) + ',' + result + '\n')
f.close()

