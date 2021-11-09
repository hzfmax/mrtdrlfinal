from utils.configs import get_configs, set_global_seeds
from ddpg.learn import learn
from itertools import chain
import numpy as np
from env import TubeEnv


def main():
    args, akwargs, ekwargs = get_configs(algo='ddpg',
                                         stochastic=False,
                                         seed=0)
    set_global_seeds(args.seed)
    def env_fn():
        return TubeEnv(**ekwargs)

    learn(env_fn, seed=args.seed, **akwargs)


if __name__ == '__main__':
    print(500)
    main()
  
