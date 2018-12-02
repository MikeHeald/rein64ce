# rein64ce
Reinforcement learning for Nintendo 64.
Currently only works with Mupen64plus 64-bit on Ubuntu with the default input plugin.

Usage:

./rein64ce /usr/games/mupen64plus /path/to/rom


todo:
feature - record states/actions - done
feature - experience playback training
feature - exploit/exploration options
feature - move reward definitions to json
feature - python tensorflow nn - done
bug - ptrace eventually hangs (deadlock?)
feature - load saved states without simulating key presses - done
feature - headless
