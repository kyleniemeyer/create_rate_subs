create\_rate\_subs
=======

This utility creates species and reaction rate subroutines for either the CPU (in C) or GPU (in CUDA C) from a Chemkin-format reaction mechanism.

Usage
-------

From the command line, use `python create_rate_subs.py proc mechname thermname` where `proc` is the processor type (e.g., `cpu` or `gpu`) and `mechname` and `thermname` are the names of the mechanism file and thermodynamic database, e.g.:

    $ python create_rate_subs.py cpu mech.dat therm.dat

You can also run `create_rate_subs` without a thermodynamic database if the information is held in the mechanism file (after the species are declared), e.g.:

        $ python create_rate_subs.py gpu mech.dat

License
-------

`create_rate_subs` is released under the modified BSD license, see LICENSE for details.

Author
------

Created by [Kyle Niemeyer](http://kyleniemeyer.com). Email address: [kyle.niemeyer@gmail.com](mailto:kyle.niemeyer@gmail.com)
