import tensorflow as tf
import numpy as np
import pickle, tqdm, os, json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

class TextGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_file = 'models/'+model_name
        self.model_trained = False

        if not os.path.exists(self.model_file):
            os.makedirs(self.model_file)
            os.makedirs(self.model_file+'/obj')
            os.makedirs(self.model_file+'/data')
            os.makedirs(self.model_file+'/results')

        elif os.path.exists(self.model_file+'/results/model_info.json'):
            with open(self.model_file+'/results/model_info.json', 'r') as info_file:
                model_info = json.load(info_file)
                self.vocab = model_info["vocab"]
                self.sequence_length = model_info["sequenceLength"]
                self.batch_size = model_info["batchSize"]
                self.model_trained = model_info["trained"]

    def read_data(self, dataset_file: str = None, banned_chars_file: str = None, save_cleaned_data: bool = False, verbose: bool = False):
        """
        Reads and cleans text from the dataset.
        :param dataset_file:      Dataset filename
        :param banned_chars_file: Banned characters list file name
        :param save_cleaned_data: Denotes whether to save the cleaned dataset
        :return str:              Cleaned dataset 
        """

        # change to default to clean_dataset.txt if exists
        if dataset_file == None:
            dataset_file = self.model_file+'/data/dataset.txt'

        if verbose: print("Reading: '"+dataset_file+"'")

        text = open(dataset_file, encoding="utf-8").read()

        if not save_cleaned_data: return text

        # Remove all links.
        clean_text = ' '.join(word for word in text.split(' ') if not 'http' in word)

        # Remove any characters from the banned_chars file, if the file exists.
        if banned_chars_file == None: banned_chars_file = self.model_file+'/data/banned_chars.txt'
        if os.path.exists(banned_chars_file):
            banned_chars = open(banned_chars_file, encoding="utf-8").read()
            clean_text = clean_text.translate(str.maketrans("", "",  banned_chars))
        
        self.vocab = ''.join(sorted(set(clean_text)))
        n_unique_chars = len(self.vocab)

        if verbose:
            print("unique_chars:", self.vocab)
            print("Number of characters:", len(clean_text))
            print("Number of unique characters:", n_unique_chars)

        if save_cleaned_data:
            with open(self.model_file+'/data/clean_dataset.txt', 'w') as clean_file:
                clean_file.write(clean_text)

        return clean_text

    def create_dataset(self, text: str, sequence_length: int = 100, batch_size: int = 128, verbose: bool = False):
        """
        Creates character dictionaries and builds sequences.
        :param text: Training text
        :param sequence_length: Sample lengths
        :param batch_size: Training batch size
        """

        # Create and save dictionaries converting characters to integers and vice versa.
        self.char2int = {c: i for i, c in enumerate(self.vocab)}
        self.int2char = {i: c for i, c in enumerate(self.vocab)}

        pickle.dump(self.char2int, open(self.model_file+'/obj/char2int.obj', 'wb'))
        pickle.dump(self.int2char, open(self.model_file+'/obj/int2char.obj', 'wb'))

        # Convert all text into integers and construct tf.data.Dataset object.
        self.encoded_text = np.array([self.char2int[c] for c in text])
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
        self.save_model_info()

    def train_model(self, epochs: int):
        """
        Trains the RNN model.
        :param epochs: Number of epochs to train for
        """

        self.create_model()

        self.model.summary()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Train and save the model.
        self.model.fit(self.dataset, steps_per_epoch=(len(self.encoded_text) - self.sequence_length) // self.batch_size, epochs=epochs)
        self.model.save(self.model_file+"/results/weights.h5")

        # Save model information
        self.model_trained = True
        self.save_model_info()

    def create_model(self):
        """
        Defines a model for use in training and text generation.
        """

        if self.model_trained:
            self.model = load_model(self.model_file+'/results/weights.h5')

        else:
            self.model = Sequential([
                LSTM(256, input_shape=(self.sequence_length, len(self.vocab)), return_sequences=True),
                Dropout(0.2),
                LSTM(256),
                Dense(len(self.vocab), activation="softmax"),
            ])

    def split_sample(self, sample):
        """
        Splits and encodes samples.
        :param sample: Text sample to split
        :return [str]: Input text and target characters
        """

        ds = tf.data.Dataset.from_tensors((sample[:self.sequence_length], sample[self.sequence_length]))

        for i in range(1, (len(sample)-1) // 2):
            input_ = sample[i: i+self.sequence_length]
            target = sample[i+self.sequence_length]
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

    def generate_text(self, seed: str, max_chars: int = 200, max_sentences: int = None, stop_char: str = None, verbose: bool = True):
        """
        Generates text from trained weights.
        :param seed:                Generation seed
        :param n_sentences:         Number of sentences to generate
        :param max_chars:           Maximum number of characters
        :param stop_char:           Stop generation after stop_char is generated
        :param model_info_filepath: Location of model information file
        :return str:                Generated text
        """

        """
        # If the model has not been trained during this execution,
        # all necessary data is defined from local files.
        if not self.model_trained:
            with open(self.model_file+'/results/model_info.json') as json_file:
                model_info = json.load(json_file)
                self.vocab = model_info['vocab']
                self.sequence_length = model_info['sequenceLength']
                self.batch_size = model_info['batchSize']
        """

        self.char2int = pickle.load(open(self.model_file+'/obj/char2int.obj', "rb"))
        self.int2char = pickle.load(open(self.model_file+'/obj/int2char.obj', "rb"))
        self.create_model()

        # Loads the trained weights.
        self.model.load_weights(self.model_file+"/results/weights.h5")

        n_sentences = 0
        generated = ""
        original_seed = seed
        for _ in tqdm.tqdm(range(max_chars), "Generating text"):

            # Create the input sequence.
            X = np.zeros((1, self.sequence_length, len(self.vocab)))
            for t, char in enumerate(seed):
                X[0, (self.sequence_length - len(seed)) + t, self.char2int[char]] = 1

            # Predicts the next character.
            predicted = self.model.predict(X, verbose=0)[0]
            next_index = np.argmax(predicted)
            next_char = self.int2char[next_index]

            if len(generated) > 0 and next_char == ' ' and generated[-1] in ".!?()[]'\"": n_sentences += 1

            # Add the character to results and shift the seed.
            generated += next_char
            seed = seed[1:] + next_char

            if (stop_char != None and next_char == stop_char) or (n_sentences == max_sentences):
                break

        if verbose:
            print("Seed:", original_seed)
            print("Generated text:")
            print(generated)

        return generated

    def save_model_info(self):
        """
        Saves model information for training.
        """

        model_info = {'modelName':      self.model_name,
                      'vocab':          self.vocab,
                      'sequenceLength': self.sequence_length,
                      'batchSize':      self.batch_size,
                      'trained':        self.model_trained
                     }

        with open(self.model_file+'/results/model_info.json', 'w') as json_file:
            json.dump(model_info, json_file, indent=4)