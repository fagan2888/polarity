
import os
import argparse

from polarity.train import build_and_save
from polarity.predict import predict_polarity

NEG_FIXTURE = os.path.abspath(
    os.path.join(os.getcwd(), "fixtures/rt-polaritydata/rt-polarity.neg")
)

POS_FIXTURE = os.path.abspath(
    os.path.join(os.getcwd(), "fixtures/rt-polaritydata/rt-polarity.pos")
)

commands = {
    "train": {
        "help" : "build and save a sentiment analysis model",
        "func": build_and_save,
        "args": {
            ("-n", "--neg"): {
                "type": str, "metavar": "PATH",
                "default": NEG_FIXTURE,
                "help": "specify the negative utterances for training",
            },
            ("-p", "--pos"): {
                "type": str, "metavar": "PATH",
                "default": POS_FIXTURE,
                "help": "specify the positive utterances for training",
            },
            ("-o", "--outpath"): {
                "type": str, "metavar": "DIR",
                "default": None,
                "help": "directory to save the model and info to",
            },
        },
    },
    "predict": {
        "help": "use sentiment analyis model to predict polarity",
        "func": predict_polarity,
        "args": {
            ("-m", "--model"): {
                "type": str, "metavar": "PKL",
                "help": "location of the pickled model on disk",
            },
            "utterances": {
                "nargs": "+",
                "help": "the utterances to predict polarity on",
            },
        },
    }
}


def main(args):
    args.func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Manages models for sentiment analysis",
        epilog="Please use GitHub issues for  help"
    )

    subparsers = parser.add_subparsers(
        title="commands", description="model management methods"
    )

    for cmd, cargs in commands.items():
        subp = subparsers.add_parser(cmd, description=cargs["help"])
        subp.set_defaults(func=cargs["func"])
        for pargs, kwargs in cargs["args"].items():
            if isinstance(pargs, str):
                pargs = (pargs,)
            subp.add_argument(*pargs, **kwargs)

    args = parser.parse_args()
    main(args)
