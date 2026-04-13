import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))

from mpx.examples.offline_task import main
import jax
jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    main(default_task="barrel_roll")
