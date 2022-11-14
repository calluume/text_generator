
# RNN Text Generator

A recurrent neural network text generator, created using tensorflow.
The project includes the files for a pre-trained text generator based on 'Alice in Wonderland,'
and a dataset of around 800,000 tweets from July and August 2020 about the US presidential election.

## Model Training

Each model is defined with a unique model name/ID, used to save all necessary files.
When the dataset is read, a ```banned_chars.txt``` file can be used to filter certain characters from the dataset before training.

```python
    # First define the model, which creates all directories.
    model = 'wonderland'
    generator = TextGenerator(model)

    # Read and encode the dataset (dataset location defaults to 'models/$MODEL$/data/dataset.txt')
    text = generator.read_data()
    generator.create_dataset(text, verbose=True)
    
    # Train the model.
    generator.train_model(30)

    # Generate text.
    random_seed = ''.join(random.choice(string.ascii_letters) for i in range(random.randint(5, 100)))
    generated_text = generator.generate_text(random_seed, max_chars=20)
```

## Text Generation

All files needed for text generation are created during the dataset creation and training, and are saved in the ```data```, ```obj``` and ```results``` files in the model's directory. These are:
 - ```weights.h5```: RNN model weights.
 - ```model_info```: Saves model info such as vocabulary, sequence length and batch size.
 - ```char2int.obj``` & ```int2char.obj```: Dictionaries for converting characters to and from integers.

Text can then be generated using the ```generate_text()``` function.

## Examples

Generating text from the 'wonderland' model:
```
Seed: rFsKZACTjOAGhFjJlYWCydvXnjmYHhuPRGAhqHUOMOczxmyAyFlvNaRCQffPEOtZv
Generated text:
e got to
in lyister! And I shall see if the next
witness! No, I'll set here! No, and do nit ear leftle golden that is I used to know. Let me
see: that it made of the glass---        OW could see's the little door!'
```

```
Seed: TtZkTtOoTaelMuRvxBBiaqmiNctcqUsAVzEToUvIr
Generated text:
goon seemer legs, you know. But do cats eat bats, I wonder what Lasity that I'm going
on she comy in lime till she'll be a presendy to seemed to be in the other
laws.

The hall, it I know she had got to things--y u inderstand will be ne time! And when I find a thing,' said the Caterpillar.

'I'm afraid I can't put it more than I am
now? That'll be a comfort, one way--near--Ho the way all store.
```
