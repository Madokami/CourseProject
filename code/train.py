import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import json

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 500

def encode_data(dataset):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(dataset.map(lambda text, label: text))
    return encoder

def encode_tweet_data(dataset):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(dataset)
    return encoder

def read_train_data():
    f = open("data/train.jsonl")
    data = []
    labels = []
    for line in f:
        jline = json.loads(line)
        context = ""
        for c in jline["context"]:
            context += c + " "
        format_line = jline["response"] + context
        data.append(format_line)
        if jline["label"] == "SARCASM":
            labels.append(1)
        else:
            labels.append(0)
    f.close()
    return (data, labels)

def read_train_data_tuple():
    f = open("data/train.jsonl")
    data = []
    for line in f:
        jline = json.loads(line)
        tuple = []
        tuple.append(jline["response"]) 
        tuple.append(jline["label"])
        data.append(tuple)
    return data

def main():
    data, labels = read_train_data()
    example = data[0]
    encoder = encode_tweet_data(data)
    vocab = np.array(encoder.get_vocabulary())
    print("vocabs")
    print(vocab[:20])
    encoded_example = encoder(example)[:3].numpy()
    print("encoded_example:")
    print(encoded_example)
    
    model = train(encoder)
    #train_dataset = tf.data.Dataset.from_tensor_slices((data[:4000], labels[:4000]))
    #test_dataset = tf.data.Dataset.from_tensor_slices((data[4001:], labels[4001:]))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['mean_squared_error'])
    #history = model.fit(train_dataset, epochs=10,
     #                   validation_data=test_dataset, 
      #                  validation_steps=30)
    history = model.fit(data, labels, epochs=10)
    model.save('/workspace/ClassificationCompetition_private/model_5')

def train(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    print([layer.supports_masking for layer in model.layers])
    return model

if __name__ == "__main__":
    main()

