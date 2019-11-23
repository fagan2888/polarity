import codecs
import pickle

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer


def build_and_save(args):
    data = [
        (instance, label)
        for path, label in [(args.neg, "neg"), (args.pos, "pos")]
        for instance in read_instances(path)
    ]

    X, y = zip(*data)
    y = LabelEncoder().fit_transform(y)

    # TODO: loop through many models and fit the best one.
    model = Pipeline([
        ("vec", CountVectorizer()),
        ("clf", MultinomialNB()),
    ])

    # Get the cross validated score
    score = cross_val_score(model, X, y, scoring="f1", cv=12)
    print(score)

    # Fit the model on all of the data and save it
    model.fit(X, y)
    with open(args.outpath, 'wb') as f:
        pickle.dump(model, f)


def read_instances(path):
    with codecs.open(path, 'r', 'latin-1') as f:
        for line in f:
            line = line.strip()
            yield line
