import logging
from testing import testing
from manual_movement import manual_movement
from manual_training import manual_training
from auto_training import auto_training
from car import CarActions


if __name__ == "__main__":
    '''Uncomment a function'''
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '\n%(asctime)s %(module)s %(levelname)s: %(message)s', datefmt = '%I:%M:%S %p', level = logging.DEBUG)
    # manual_movement('simpler')
    # auto_training(50000, model_only=False)
    # testing('episode_1060')
    # manual_training(CarActions.FORWARD.value)