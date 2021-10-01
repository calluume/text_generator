import tensorflow as tf
import numpy as np
import pickle, tqdm, os, json, string, random
from tensorflow import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class TextGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_trained = False

    def read_data(self, filename, save_cleaned_data=False, verbose=False):
        """
        Reads and cleans text from the dataset.
        :param filename:          Dataset filename
        :param save_cleaned_data: Denotes whether to save the cleaned dataset
        :return str:              Cleaned dataset 
        """

        if verbose: print("Reading: '"+filename+"'")

        text = open(filename, encoding="utf-8").read()

        # Remove all links.
        clean_text = ' '.join(word for word in text.split(' ') if not 'http' in word)

        # Remove any characters from the banned_chars file, if the file exists.
        if os.path.exists('data/banned_chars.txt'):
            banned_chars = open('data/banned_chars.txt', encoding="utf-8").read()
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
        """
        Creates character dictionaries and builds sequences.
        :param text: Training text
        :param sequence_length: Sample lengths
        :param batch_size: Training batch size
        """

        # Dictionary converting characters to integers.
        self.char2int = {c: i for i, c in enumerate(self.vocab)}
        # Dictionary converting integers to characters.
        self.int2char = {i: c for i, c in enumerate(self.vocab)}

        pickle.dump(self.char2int, open("obj/"+self.model_name+"_char2int.obj", "wb"))
        pickle.dump(self.int2char, open("obj/"+self.model_name+"_int2char.obj", "wb"))

        # Convert all text into integers.
        self.encoded_text = np.array([self.char2int[c] for c in text])
        # Construct tf.data.Dataset object.
        char_dataset = tf.data.Dataset.from_tensor_slices(self.encoded_text)
        
        # Build sequences by batching.
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        sequences = char_dataset.batch(2*self.sequence_length + 1, drop_remainder=True)

        self.dataset = sequences.flat_map(self.split_sample).map(self.one_hot_samples)

        # Prints the first 2 samples.
        if verbose:
            for element in self.dataset.take(2):
                print("Input:", ''.join([self.int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
                print("Target:", self.int2char[np.argmax(element[1].numpy())])
                print("Input shape:", element[0].shape)
                print("Target shape:", element[1].shape)
                print("="*50, "\n")

        # Repeats, shuffles and batches the dataset.
        self.dataset = self.dataset.repeat().shuffle(1024).batch(self.batch_size, drop_remainder=True)

    def train_model(self, epochs):
        """
        Trains the RNN model.
        :param epochs: Number of epochs to train for
        """

        self.create_model()

        self.model.summary()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Make results folder if does not exist yet.
        if not os.path.isdir("results"):
            os.mkdir("results")

        # Train and save the model.
        self.model.fit(self.dataset, steps_per_epoch=(len(self.encoded_text) - self.sequence_length) // self.batch_size, epochs=epochs)
        self.model.save("results/"+self.model_name+"_weights.h5")

        # Saves model information as a json file for text generation.
        model_info = {'modelName':      self.model_name,
                      'vocab':          self.vocab,
                      'sequenceLength': self.sequence_length,
                      'batchSize':      self.batch_size}

        with open('data/'+self.model_name+'_model_info.json', 'w') as json_file:
            json.dump(model_info, json_file, indent=4)

        # Tells the model that it doesn't need to load model data
        # from local files.
        self.model_trained = True

    def create_model(self):
        """
        Defines a model for use in training and text generation.
        """

        self.model = Sequential([
            LSTM(256, input_shape=(self.sequence_length, len(self.vocab)), return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dense(len(self.vocab), activation="softmax"),
        ])

    def split_sample(self, sample):
        """
        Splits and encodes samples.
        :param sample: Text sample to split
        :return [str]: Input text and target characters
        """

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
        """
        Encodes the inputs and targets asbinary vectors
        :param input:  Sample
        :param target: Target variable
        :return [0|1]: Encoded variable
        """

        # Example:
        # if character 'd' is encoded as 3 and n_unique_chars = 5
        # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
        return tf.one_hot(input_, len(self.vocab)), tf.one_hot(target, len(self.vocab))

    def generate_text(self, seed, n_sentences=1, max_chars=400, model_file=None):
        """
        Generates text from trained weights.
        :param seed:        Generation seed
        :param n_sentences: Number of sentences to generate
        :param max_chars:   Maximum number of characters
        :param model_file:  Location of model information file
        :return str:        Generated text
        """

        if model_file == None:
            model_file = self.model_name+'_model_info.json'

        # If the model has not been trained during this execution,
        # all necessary data is defined from local files.
        if not self.model_trained:
            with open('data/'+model_file) as json_file:
                model_info = json.load(json_file)
                self.vocab = model_info['vocab']
                self.sequence_length = model_info['sequenceLength']
                self.batch_size = model_info['batchSize']

            self.char2int = pickle.load(open("obj/"+self.model_name+"_char2int.obj", "rb"))
            self.int2char = pickle.load(open("obj/"+self.model_name+"_int2char.obj", "rb"))
            self.create_model()

        vocab_size = len(self.char2int)

        # Loads the trained weights.
        self.model.load_weights("results/"+self.model_name+"_weights.h5")

        generated = ""
        original_seed = seed
        for i in tqdm.tqdm(range(max_chars), "Generating text"):
            # Create the input sequence.
            X = np.zeros((1, self.sequence_length, len(self.vocab)))
            for t, char in enumerate(seed):
                X[0, (self.sequence_length - len(seed)) + t, self.char2int[char]] = 1
            # Predicts the next character.
            predicted = self.model.predict(X, verbose=0)[0]
            # Converting the vector to an integer.
            next_index = np.argmax(predicted)
            # Converting the integer to a character.
            next_char = self.int2char[next_index]
            # Adding the character to results.
            generated += next_char
            # Shifts seed and the predicted character.
            seed = seed[1:] + next_char

        print("Seed:", original_seed)
        print("Generated text:")
        print(generated)

        return generated

if __name__ == "__main__":
    # Define the model.
    model = 'wonderland' # 'tweets' or 'wonderland'
    generator = TextGenerator(model)

    # Create the dataset and train the model.
    #text = generator.read_data('data/'+model+'.txt', True, verbose=True)
    #generator.create_dataset(text, verbose=True)
    #generator.train_model(30)

    # Generates text.
    random_seed = ''.join(random.choice(string.ascii_letters) for i in range(random.randint(5, 100)))
    generated_text = generator.generate_text(random_seed)