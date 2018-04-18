from util.config import print_usage, get_config
from src.RPSMNetwork import RPSMNetwork
from src.h5_dataset import H5Dataset

#Machine Learning Project Using Tensorflow for Convolutional Neural Networks
def main(config):
    """The main function."""
    h5_train = H5Dataset(config, 'train')
    h5_val = H5Dataset(config, 'valid')
    mynet = RPSMNetwork(h5_train, h5_val, config)

    mynet.train(config)


if __name__ == "__main__":
    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
