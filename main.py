from util.config import print_usage, get_config
from src.RPSMNetwork import RPSMNetwork
from src.h5_dataset import H5Dataset


def main(config):
    """The main function."""
    mynet = RPSMNetwork(config)
    h5_train = H5Dataset(config, 'train')
    h5_val = H5Dataset(config, 'valid')
    x_tr, y_tr = h5_train.get()
    x_va, y_va = h5_val.get()

    mynet.train(x_tr, y_tr, x_va, y_va)


if __name__ == "__main__":
    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
