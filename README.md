# Polarity

Example of a CLI application to detect sentence polarity based on movie reviews.

## Getting Started

First, make sure that you unzip the `rt-polaritydata.zip` file in the `fixtures` folder. E.g.

    $ cd fixtures
    $ unzip rt-polaritydata.zip
    $ cd ..

You should now have the following files:

- `fixtures/rt-polaritydata.README.1.0.txt`
- `fixtures/rt-polaritydata/rt-polarity.neg`
- `fixtures/rt-polaritydata/rt-polarity.pos`

You can run the polarity script as follows:

    $ python -m polarity --help

### Training Models

Train a model as follows:

    $ python -m polarity train -o fixtures/mymodel.pickle

This trains a model and prints out the scores, saving it to the path specified by the `-o` flag.

#### Next Steps with Training

Adapt the training script as follows:

1. Perform feature engineering: try different word features including TF-IDF, one-hot encoding, feature hashes, word embeddings, etc. Try removing stopwords, punctuation, low frequency words etc.
2. Run cross-validation on a number of different algorithms and select the best model.
3. Hyperparameter tune the selected best-model using GridSearch.
4. Save more information about the trained model in a JSON file, e.g. the number of instances, the cross-validation scores, the date trained, etc.
5. Save yellowbrick visualizations that describe how the model is performing.
6. Specify a default location for the model to be saved, e.g. "fixtures/modelname-YYYYMMDDHHMM.pkl", where model name is the class (e.g. `MultinomialNB` and YYYYMMDDHHMM is the current datetime).
7. Implement a progress bar that gives the user feedback about how training is proceeding.

### Predicting on Text

Once you have a trained model start predicting on text as follows:

    $ python -m polarity predict -m fixtures/mymodel.pickle "This movie was very bad"

Please note that you have to submit each sentence surrounded by quotation marks (`""`) and that you can enter multiple sentences to be predicted on. Also note that tokenization is split by space, including punctuation.

#### Next Steps with Predicting

1. Ensure the label encoder can be used to specify "neg" and "pos" instead of zero and one. 
2. Ensure that the probabilities for each class and the selected class are reported.


### Other next Steps

1. Use nltk to tokenize words and sentences
2. Feed more information into the model such as part of speech tags or grammar indicators
3. Create an option to report "most informative features" on the command line
4. Allow the predictor to read from stdin to classify a file or other input
