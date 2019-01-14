# rein64ce

Reinforcement learning for Nintendo 64.

Currently only works with Mupen64plus 64-bit on Ubuntu with the default input plugin.

Usage:

./rein64ce /usr/games/mupen64plus /path/to/rom


TODO:

feature - marshal agent setting and NN weights to json for saving/sharing

feature - record states/actions - done

feature - exploit/exploration options -- done -- added epsilon-greedy and boltzmann

feature - move reward definitions to json

feature - python tensorflow nn - done

bug - ptrace eventually hangs (deadlock?)

feature - load saved states without simulating key presses - done

