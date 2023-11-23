#!/usr/bin/env python3
import sys
import getopt

from config import get_config

from tutorial1 import train_model1

def main(argv):
    config_filename = None
    model_folder = None
    try:
        opts, args = getopt.getopt(argv, "hc:m:", ["config=", "modelfolder="])
    except getopt.GetoptError:
        print('train.py -c <config_file> -m <model_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -c <config_file> -m <model_folder>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_filename = arg
        elif opt in ("-m", "--modelfolder"):
            model_folder = arg

    # warnings.filterwarnings('ignore')
    config = get_config(config_filename, model_folder)

    match config['alt_model']:
        case "model1":
            train_model1(config)
        case _:
            train_model1(config)


if __name__ == "__main__":
    main(sys.argv[1:])
