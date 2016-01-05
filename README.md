create\_rate\_subs
=======

This utility creates species and reaction rate subroutines for either C or CUDA from a Chemkin- or Cantera-format reaction mechanism.

Usage
-------

`create\_rate\_subs` can be run either as an executable or script via Python. To run as an executable, from the command line change to the proper directory, change the file mode to executable and run:

    chmod +x create_rate_subs.py
    ./create_rate_subs [options]

To run it as a script, change to the appropriate directory and run:

    python create_rate_subs.py [options]

The generated source code is placed within the `out` directory, which is created if it doesn't exist initially.

Options
-------

In the above, `[options]` indicates where command line options should be specified. The options available can be seen using `-h` or `--help`, or below:

    -h, --help            show this help message and exit
    -l {c,cuda,fortran,matlab}, --lang {c,cuda,fortran,matlab}
                          Programming language for output source files.
    -i INPUT, --input INPUT
                          Input mechanism filename (e.g., mech.dat).
    -t THERMO, --thermo THERMO
                          Thermodynamic database filename (e.g., therm.dat), or
                          nothing if in mechanism.
    -ls LAST_SPECIES, --last_species LAST_SPECIES
                          The name of the species to set as the last in the
                          mechanism. If not specified, defaults to the first of
                          N2, AR, and HE in the mechanism.


License
-------

`create_rate_subs` is released under the modified BSD license, see LICENSE for details.

If you use this package as part of a scholarly publication, please cite the following paper in addition to this resource:

 * KE Niemeyer and CJ Sung. Accelerating moderately stiff chemical kinetics in reactive-flow simulations using GPUs. *J. Comput. Phys.*, 256:854-871, 2014. doi:[10.1016/j.jcp.2013.09.025](http://dx.doi.org/10.1016/j.jcp.2013.09.025)

Author
------

Created by [Kyle Niemeyer](http://kyleniemeyer.com). Email address: [kyle.niemeyer@gmail.com](mailto:kyle.niemeyer@gmail.com)
