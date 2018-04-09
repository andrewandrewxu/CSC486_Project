from src.machine import Machine
from util.config import print_usage, get_config


def main(config):
    """The main function."""
    mynet = Machine(config)
    mynet.train()
    mynet.test()


if __name__ == "__main__":
    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
