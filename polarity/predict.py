import pickle


def predict_polarity(args):
    if args.model is None:
        raise ValueError("please specify the location of the model")

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    yhat = model.predict_proba(args.utterances)
    for i, label in enumerate(yhat):
        print(f"{i}: {label}")
