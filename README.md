# Game Theory methods
This is a repository containing didactic material for a first course in Game Theory building on top on [nashpy](https://nashpy.readthedocs.io/en/stable/). This is beta version so use it with care. If you find issues/bugs please share with us.

## Installation
Follow these steps to set up the environment and install the necessary dependencies.

### Cloning the Repository
Clone the repository and navigate to the project directory:

```
$ git clone https://github.com/JuanViguera/game-theory
$ cd game-theory
```

### Installing Dependencies
We recommend to create an environment using [mamba](https://mamba.readthedocs.io/en/latest/index.html) or a similar package manager. 
After creating an environment containing python (including pip), install the required dependencies by running:

```
$ make install
```
If you plan to use jupyter notebooks, you can create kernel by running:

```
$ python -m ipykernel install --user --name gt-env
```

## Usage
See [demo.ipynb](./demo.ipynb) for some examples of functionalities of this repository.
