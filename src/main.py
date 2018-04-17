import numpy as np
from src.RPSMNetwork import RPSMNetwork
from util.config import print_usage, get_config


def main(config):
    """The main function."""
    mynet = RPSMNetwork(config)
    x_tr = np.zeros((1, 16, 368, 368, 3))
    y_tr = np.zeros((1, 16, 51))
    x_va = np.zeros((1, 16, 368, 368, 3))
    y_va = np.zeros((1, 16, 51))

    mynet.train(x_tr, y_tr, x_va, y_va)

    x_te = np.zeros((1, 16, 376, 376, 3))
    y_te = np.zeros((1, 16, 51))

    #mynet.test(x_te, y_te)


if __name__ == "__main__":
    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
