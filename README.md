# car_racing

Car_racing is a simple Python racing game with Q Learning caveat.

## Installation

Use the `environment.yml` file to create a [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment containing all needed packages and activate it.

```bash
conda env create -f requirements.yml
conda activate qlearning
```

This is all you have to do in order to run this app. Though recommended step is to create `memory` and `models` directories.

```bash
cd car_racing
mkdir memory
mkdir models
```

## Usage

Recommended way to run `car_racing` is to modify the `main.py` file. It contains the standart `if __name__=='__main__:` if clause that consists of logger setup and 4 commented lines.

```python
if __name__ == "__main__":
    '''Uncomment a function'''
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s %(module)s %(levelname)s: %(message)s', datefmt = '%I:%M:%S %p', level = logging.DEBUG)
    # manual_movement('simpler')
    # auto_training(50000, model_only=False)
    # testing('episode_1060')
    # manual_training(CarActions.FORWARD.value)
```

As the comment suggests, easiest way for you is to uncomment one of the lines and then run the file by

```bash
cd car_racing
python main.py
```

Though be wary of the `testing` function, it requires a trained neural network in the `models` folder.

## License

[MIT](https://choosealicense.com/licenses/mit/)
