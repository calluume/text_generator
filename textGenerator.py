import tensorflow as tf
import numpy as np
import pickle, tqdm, os, json
from tensorflow import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout#, Activation
#from tensorflow.keras.callbacks import ModelCheckpoint
from string import punctuation

class TextGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_trained = False

    def read_data(self, filename, save_cleaned_data=True, verbose=False):
        if verbose: print("Reading: '"+filename+"'")

        text = open(filename, encoding="utf-8").read()
        banned_chars = open('data/banned_chars.txt', encoding="utf-8").read()

        # Remove all links
        clean_text = ' '.join(word for word in text.split(' ') if not 'http' in word)
        # Remove all banned chars
        clean_text = clean_text.translate(str.maketrans("", "",  banned_chars))
        
        self.vocab = ''.join(sorted(set(clean_text)))
        n_unique_chars = len(self.vocab)

        if verbose:
            print("unique_chars:", self.vocab)
            print("Number of characters:", len(clean_text))
            print("Number of unique characters:", n_unique_chars)

        if save_cleaned_data:
            with open(filename.split('.')[0]+'_cleaned.txt', 'w') as clean_file:
                clean_file.write(clean_text)
        return clean_text

    def create_dataset(self, text, sequence_length=100, batch_size=128, verbose=False):
        # dictionary that converts characters to integers
        self.char2int = {c: i for i, c in enumerate(self.vocab)}
        # dictionary that converts integers to characters
        self.int2char = {i: c for i, c in enumerate(self.vocab)}

        # save these dictionaries for later generation
        pickle.dump(self.char2int, open("obj/"+self.model_name+"_char2int.obj", "wb"))
        pickle.dump(self.int2char, open("obj/"+self.model_name+"_int2char.obj", "wb"))

        # convert all text into integers
        self.encoded_text = np.array([self.char2int[c] for c in text])
        # construct tf.data.Dataset object
        char_dataset = tf.data.Dataset.from_tensor_slices(self.encoded_text)
        
        # build sequences by batching
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        sequences = char_dataset.batch(2*self.sequence_length + 1, drop_remainder=True)

        self.dataset = sequences.flat_map(self.split_sample).map(self.one_hot_samples)

        # print first 2 samples
        if verbose:
            for element in self.dataset.take(2):
                print("Input:", ''.join([self.int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
                print("Target:", self.int2char[np.argmax(element[1].numpy())])
                print("Input shape:", element[0].shape)
                print("Target shape:", element[1].shape)
                print("="*50, "\n")

        # repeat, shuffle and batch the dataset
        self.dataset = self.dataset.repeat().shuffle(1024).batch(self.batch_size, drop_remainder=True)

    def train_model(self, epochs):
        self.create_model()

        self.model.summary()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # make results folder if does not exist yet
        if not os.path.isdir("results"):
            os.mkdir("results")
        # train the model
        self.model.fit(self.dataset, steps_per_epoch=(len(self.encoded_text) - self.sequence_length) // self.batch_size, epochs=epochs)
        # save the model
        self.model.save("results/"+self.model_name+"_weights.h5")

        model_info = {'modelName':      self.model_name,
                      'vocab':          self.vocab,
                      'sequenceLength': self.sequence_length,
                      'batchSize':      self.batch_size}

        with open('data/'+self.model_name+'_model_info.json', 'w') as json_file:
            json.dump(model_info, json_file, indent=4)

        self.model_trained = True

    def create_model(self):
        # a better model (slower to train obviously)
        self.model = Sequential([
            LSTM(256, input_shape=(self.sequence_length, len(self.vocab)), return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dense(len(self.vocab), activation="softmax"),
        ])

    def split_sample(self, sample):
        # example :
        # sequence_length is 10
        # sample is "python is a great pro" (21 length)
        # ds will equal to ('python is ', 'a') encoded as integers
        
        ds = tf.data.Dataset.from_tensors((sample[:self.sequence_length], sample[self.sequence_length]))
        for i in range(1, (len(sample)-1) // 2):
            # first (input_, target) will be ('ython is a', ' ')
            # second (input_, target) will be ('thon is a ', 'g')
            # third (input_, target) will be ('hon is a g', 'r')
            # and so on
            input_ = sample[i: i+self.sequence_length]
            target = sample[i+self.sequence_length]
            # extend the dataset with these samples by concatenate() method
            other_ds = tf.data.Dataset.from_tensors((input_, target))
            ds = ds.concatenate(other_ds)
        return ds

    def one_hot_samples(self, input_, target):
        # onehot encode the inputs and the targets
        # Example:
        # if character 'd' is encoded as 3 and n_unique_chars = 5
        # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
        return tf.one_hot(input_, len(self.vocab)), tf.one_hot(target, len(self.vocab))

    def generate_text(self, seed, n_sentences=1, max_chars=400, model_file='example.json'):

        if not self.model_trained:
            with open('data/'+model_file) as json_file:
                model_info = json.load(json_file)
                self.model_name = model_info['modelName']
                self.vocab = model_info['vocab']
                self.sequence_length = model_info['sequenceLength']
                self.batch_size = model_info['batchSize']

            self.char2int = pickle.load(open("obj/"+self.model_name+"_char2int.obj", "rb"))
            self.int2char = pickle.load(open("obj/"+self.model_name+"_int2char.obj", "rb"))
            self.create_model()

        vocab_size = len(self.char2int)

        # load the optimal weights
        self.model.load_weights("results/"+self.model_name+"_weights.h5")

        generated = ""
        original_seed = seed
        for i in tqdm.tqdm(range(max_chars), "Generating text"):
            # make the input sequence
            X = np.zeros((1, self.sequence_length, len(self.vocab)))
            for t, char in enumerate(seed):
                X[0, (self.sequence_length - len(seed)) + t, self.char2int[char]] = 1
            # predict the next character
            predicted = self.model.predict(X, verbose=0)[0]
            # converting the vector to an integer
            next_index = np.argmax(predicted)
            # converting the integer to a character
            next_char = self.int2char[next_index]
            # add the character to results
            generated += next_char
            # shift seed and the predicted character
            seed = seed[1:] + next_char

        print("Seed:", original_seed)
        print("Generated text:")
        print(generated)

if __name__ == "__main__":
    
    generator = TextGenerator('tweets')
    text = generator.read_data('data/tweets.txt', True, verbose=True)
    generator.create_dataset(text, verbose=True)
    generator.train_model(3)
    generator.generate_text("Donald Trump", 100, 'tweets_model_info.json')