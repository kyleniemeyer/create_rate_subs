#! /usr/bin/env python
"""Creates source code for species production rate calculation."""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math
from argparse import ArgumentParser

# Local imports
import chem_utilities as chem
import mech_interpret as mech
import rate_subs as rate
import utils


def write_main(path, lang, specs):
    """Writes sample main file.
    """
    if lang == 'c':
        filename = 'main_cpu.c'
        pre = ''
    elif lang == 'cuda':
        filename = 'main_gpu.cu'
        pre = '__global__ '
    elif lang == 'fortran':
        filename = 'main.f90'
        pre = ''
    elif lang == 'matlab':
        filename = 'main.m'
        pre = ''

    num_eq = len(specs) + 1

    file = open(path + filename, 'w')

    # include other subroutine files
    file.write('#include "header.h"\n\n')

    if lang == 'c':
        file.write('#include "chem_utils.h"\n')
        file.write('#include "rates.h"\n')
    elif lang == 'cuda':
        file.write('/** CUDA libraries */\n')
        file.write('#include <cuda.h>\n')
        file.write('#include <cutil.h>\n')
        file.write('\n')

        file.write('#include "chem_utils_gpu.cuh"\n')
        file.write('#include "rates.cuh"\n')

    file.write('///////////////////////////////////////////////////////\n\n')

    file.write(pre + 'void intDriver (const Real t, const Real h, '
              'const Real pr, Real* y_global ) {\n\n')

    if lang == 'c':
        file.write('  // loop over all "threads"\n')
        file.write('  for ( uint tid = 0; tid < NUM; ++tid ) {\n\n')
        tab = '    '
    elif lang == 'cuda':
        file.write('  // unique thread ID, based on local '
                   'ID in block and block ID\n')
        file.write('  uint tid = threadIdx.x + '
                   '( blockDim.x * blockIdx.x )' +
                   utils.line_end[lang] + '\n')
        tab = '  '

    file.write(tab + '// local array with initial values\n')
    file.write(tab + 'Real y0_local[' + str(num_eq) + ']' +
               utils.line_end[lang])
    file.write(tab + '// local array with integrated values\n')
    file.write(tab + 'Real yn_local[' + str(num_eq) + ']' +
               utils.line_end[lang] + '\n')

    file.write(tab + '// load local array with initial '
               'values from global array\n')
    for i in xrange(num_eq):
        line = (tab + 'y0_local[' + str(i) + '] = '
                'y_global[tid + NUM * ' + str(i) + ']' +
                utils.line_end[lang])
        file.write(line)
    file.write('\n')

    file.write(tab + '// call integrator for one time step\n')
    file.write(tab + 'INTEGRATOR ( t, pr, h, y0_local, yn_local )' +
               utils.line_end[lang] + '\n')

    file.write(tab + '// update global array with integrated values\n')
    for i in xrange(num_eq):
        line = (tab + 'y_global[tid + NUM * ' + str(i) + '] = '
                'yn_local[' + str(i) + ']' + utils.line_end[lang])
        file.write(line)
    file.write('\n')

    if lang == 'c':
        file.write('  } // end tid loop\n\n')

    file.write('} // end intDriver\n\n')

    file.write('///////////////////////////////////////////////////////\n\n')

    file.write('int main ( void ) {\n\n')

    if lang == 'c':
        file.write('  // print number of threads\n' +
                   '  printf ("# threads: %d\\n", NUM)' +
                   utils.line_end[lang] + '\n')
    else:
        file.write('  // print number of threads and block size\n' +
                   '  printf ("# threads: %d \\t block size: %d\\n", '
                   'NUM, BLOCK)' + utils.line_end[lang] + '\n')

    file.write('  // starting time (usually 0.0), units [s]\n')
    file.write('  Real t0 = 0.0' + utils.line_end[lang])
    file.write('  // ending time of integration, units [s]\n')
    file.write('  Real tend = 1.0e-7' + utils.line_end[lang])
    file.write('  // time step size, units [s]\n')
    file.write('  Real h = 1.0e-8' + utils.line_end[lang])
    file.write('  // number of steps, based on time range and step size\n')
    file.write('  uint steps = (tend - t0)/h' + utils.line_end[lang] + '\n')

    file.write('  // species indices:\n')
    for sp in specs:
        file.write('  // ' + str(specs.index(sp)) + ' ' + sp.name + '\n')
    file.write('\n')

    file.write('  // initial mole fractions\n')
    file.write('  Real Xi[{:}]'.format(num_eq - 1) + utils.line_end[lang])
    file.write('  for ( int j = 0; j < {:}; ++ j ) {{\n'.format(num_eq - 1))
    file.write('    Xi[j] = 0.0' + utils.line_end[lang])
    file.write('  }\n')
    file.write('\n')

    file.write('  //\n  // set initial mole fractions here\n  //\n\n')
    file.write('  // normalize mole fractions to sum to 1\n')
    file.write('  Real Xsum = 0.0' + utils.line_end[lang])
    file.write('  for ( int j = 0; j < {:}; ++ j ) {{\n'.format(num_eq - 1))
    file.write('    Xsum += Xi[j]' + utils.line_end[lang])
    file.write('  }\n')
    file.write('  for ( int j = 0; j < {:}; ++ j ) {{\n'.format(num_eq - 1))
    file.write('    Xi[j] /= Xsum' + utils.line_end[lang])
    file.write('  }\n\n')

    file.write('  // initial mass fractions\n')
    file.write('  Real Yi[{:}]'.format(num_eq - 1) + utils.line_end[lang])
    file.write('  mole2mass ( Xi, Yi )' + utils.line_end[lang] + '\n')

    file.write('  // size of data array in bytes\n')
    file.write('  uint size = NUM * sizeof(Real) * {:}' +
               utils.line_end[lang] + '\n'.format(num_eq))

    file.write('  // pointer to data on host memory\n')
    file.write('  Real *y_host' + utils.line_end[lang])
    file.write('  // allocate memory for all data on host\n')
    file.write('  y_host = (Real *) malloc (size)' +
               utils.line_end[lang] + '\n')

    file.write('  // set initial pressure, units [dyn/cm^2]\n')
    file.write('  // 1 atm = 1.01325e6 dyn/cm^2\n')
    file.write('  Real pres = 1.01325e6' + utils.line_end[lang])
    file.write('  // set initial temperature, units [K]\n')
    file.write('  Real T0 = 1600.0' + utils.line_end[lang])

    file.write('  // load temperature and mass fractions for all threads (cells)\n')
    file.write('  for ( int i = 0; i < NUM; ++i ) {\n')
    file.write('    y_host[i] = T0;\n')
    file.write('    // loop through species\n')
    file.write('    for ( int j = 1; j < {:}; ++j) {{\n'.format(num_eq))
    file.write('      y_host[i + NUM * j] = Yi[j - 1];\n')
    file.write('    }\n')
    file.write('  }\n\n')

    file.write('#ifdef CONV\n')
    file.write('  // if constant volume, calculate density\n')
    file.write('  Real rho = 0.0;\n')
    for sp in specs:
        file.write('  rho += Xi[{:}] * {:};\n'.format(specs.index(sp), sp.mw))
    file.write('  rho = pres * rho / ( {:} * T0 );\n'.format(chem.RU))
    file.write('#endif\n\n')

    file.write('#ifdef IGN\n')
    file.write('  // flag for ignition\n')
    file.write('  bool ign_flag = false;\n')
    file.write('  // ignition delay time, units [s]\n')
    file.write('  Real t_ign = 0.0;\n')
    file.write('#endif\n\n')

    file.write('  // set time to initial time\n')
    file.write('  Real t = t0;\n\n')

    if lang == 'c':
        file.write('  // timer start point\n')
        file.write('  clock_t t_start;\n')
        file.write('  // timer end point\n')
        file.write('  clock_t t_end;\n\n')

        file.write('  // start timer\n')
        file.write('  t_start = clock();\n\n')
    elif lang == 'cuda':
        file.write('  // set GPU card to one other than primary\n')
        file.write('  cudaSetDevice (1);\n\n')

        file.write('  // integer holding timer time\n')
        file.write('  uint timer_compute = 0;\n\n')
        file.write('  // create timer object\n')
        file.write('  CUT_SAFE_CALL ( cutCreateTimer ( &timer_compute ) );\n')
        file.write('  // start timer\n')
        file.write('  CUT_SAFE_CALL ( cutStartTimer ( timer_compute ) );\n\n')

    file.write('  // pointer to memory used for integration\n')
    file.write('  Real *y_device;\n')

    if lang == 'c':
        file.write('  // allocate memory\n')
        file.write('  y_device = (Real *) malloc ( size );\n\n')
    elif lang == 'cuda':
        file.write('  // allocate memory on device\n')
        file.write('  CUDA_SAFE_CALL ( cudaMalloc ( (void**) &y_device, size ) );\n\n')

    # time integration loop
    file.write('  // time integration loop\n')
    file.write('  while ( t < tend ) {\n\n')
    if lang == 'c':
        file.write('    // copy local array to "global" array\n')
        file.write('    memcpy ( y_device, y_host, size );\n\n')

        file.write('#if defined(CONP)\n')
        file.write('    // constant pressure case\n')
        file.write('    intDriver ( t, h, pres, y_device );\n')
        file.write('#elif defined(CONV)\n')
        file.write('    // constant volume case\n')
        file.write('    intDriver ( t, h, rho, y_device );\n')
        file.write('#endif\n\n')

        file.write('    // transfer integrated data back to local array\n')
        file.write('    memcpy ( y_host, y_device, size );\n\n')
    elif lang == 'cuda':
        file.write('    // copy data on host to device\n')
        file.write('    CUDA_SAFE_CALL ( cudaMemcpy ( y_device, y_host, size, cudaMemcpyHostToDevice ) );\n\n')
        file.write('    //\n    // kernel invocation\n    //\n\n')
        file.write('    // block size\n')
        file.write('    dim3 dimBlock ( BLOCK, 1 );\n')
        file.write('    // grid size\n')
        file.write('    dim3 dimGrid ( NUM / BLOCK, 1 );\n\n')

        file.write('#if defined(CONP)\n')
        file.write('    // constant pressure case\n')
        file.write('    intDriver <<< dimGrid, dimBlock >>> ( t, h, pres, y_device );\n')
        file.write('#elif defined(CONV)\n')
        file.write('    // constant volume case\n')
        file.write('    intDriver <<< dimGrid, dimBlock >>> ( t, h, rho, y_device );\n')
        file.write('#endif\n\n')

        file.write('#ifdef DEBUG\n')
        file.write('    // barrier thread synchronization\n')
        file.write('    CUDA_SAFE_CALL ( cudaThreadSynchronize() );\n')
        file.write('#endif\n\n')

        file.write('    // transfer integrated data from device back to host\n')
        file.write('    CUDA_SAFE_CALL ( cudaMemcpy ( y_host, y_device, size, cudaMemcpyDeviceToHost ) );\n\n')

    # check for ignition
    file.write('#ifdef IGN\n')
    file.write('    // determine if ignition has occurred\n')
    file.write('    if ( ( y_host[0] >= (T0 + 400.0) ) && !(ign_flag) ) {\n')
    file.write('      ign_flag = true;\n')
    file.write('      t_ign = t;\n')
    file.write('    }\n')
    file.write('#endif\n\n')

    file.write('    // increase time by one step\n')
    file.write('    t += h;\n\n')
    file.write('  } // end time loop\n\n')

    # after integration, free memory and stop timer
    if lang == 'c':
        file.write('  // free data array from global memory\n')
        file.write('  free ( y_device );\n\n')

        file.write('  // stop timer\n')
        file.write('  t_end = clock();\n\n')

        file.write('  // get clock tiem in seconds\n')
        file.write('  Real tim = ( t_end - t_start ) / ( (Real)(CLOCKS_PER_SEC) );\n')
    elif lang ==  'cuda':
        file.write('  // free data array from device memory\n')
        file.write('  CUDA_SAFE_CALL ( cudaFree ( y_device ) );\n\n')

        file.write('  // stop timer\n')
        file.write('  CUT_SAFE_CALL ( cutStopTimer ( timer_compute ) );\n\n')

        file.write('  // get clock time in seconds; cutGetTimerValue() returns ms\n')
        file.write('  Real tim = cutGetTimerValue ( timer_compute ) / 1000.0;\n')
    file.write('  tim = tim / ( (Real)(steps) );\n')

    # print time
    file.write('  // print time per step and time per step per thread\n')
    file.write('  printf("Compute time per step: %e (s)\\t%e (s/thread)\\n", tim, tim / NUM);\n\n')

    file.write('#ifdef CONV\n')
    file.write('  // calculate final pressure for constant volume case\n')
    file.write('  pres = 0.0;\n')
    for sp in specs:
        file.write('  pres += y_host[1 + NUM * {:}] / {:};\n'.format(specs.index(sp), sp.mw))
    file.write('  pres = rho * {:} * y_host[0] * pres;\n'.format(chem.RU))
    file.write('#endif\n\n')

    file.write('#ifdef DEBUG\n')
    file.write('  // if debugging/testing, print temperature and first species mass fraction of last thread\n')
    file.write('  printf ("T[NUM-1]: %f, Yh: %e\\n", y_host[NUM-1], y_host[NUM-1+NUM]);\n')
    file.write('#endif\n\n')

    file.write('#ifdef IGN\n')
    file.write('  // if calculating ignition delay, print ign delay; units [s]\n')
    file.write('  printf ( "Ignition delay: %le\\n", t_ign );\n')
    file.write('#endif\n\n')

    file.write('  // free local data array\n')
    file.write('  free ( y_host );\n\n')

    file.write('  return 0;\n')
    file.write('} // end main\n')

    file.close()
    return


def write_header(path, lang, specs):
    """Writes C header file.
    """
    nsp = len(specs)
    num_eq = nsp + 1

    file = open(path + 'header.h', 'w')

    file.write('#include <stdlib.h>\n')
    file.write('#include <stdio.h>\n')
    file.write('#include <assert.h>\n')
    file.write('#include <time.h>\n')
    file.write('#include <math.h>\n')
    file.write('#include <string.h>\n')
    file.write('#include <stdbool.h>\n')
    file.write('\n')

    file.write('/** number of threads */\n')
    file.write('#define NUM 65536\n')
    file.write('/** GPU block size */\n')
    file.write('#define BLOCK 128\n')
    file.write('\n')

    file.write(
    '/** Sets precision as double or float. */\n' +
    '#define DOUBLE\n' +
    '#ifdef DOUBLE\n' +
    '  /** Define Real as double. */\n' +
    '  #define Real double\n' +
    '\n' +
    '  /** Double precision ONE. */\n' +
    '  #define ONE 1.0\n' +
    '  /** Double precision TWO. */\n' +
    '  #define TWO 2.0\n' +
    '  /** Double precision THREE. */\n' +
    '  #define THREE 3.0\n' +
    '  /** Double precision FOUR. */\n' +
    '  #define FOUR 4.0\n' +
    '#else\n' +
    '  /** Define Real as float. */\n' +
    '  #define Real float\n' +
    '\n' +
    '  /** Single precision ONE. */\n' +
    '  #define ONE 1.0f\n' +
    '  /** Single precision (float) TWO. */\n' +
    '  #define TWO 2.0f\n' +
    '  /** Single precision THREE. */\n' +
    '  #define THREE 3.0f\n' +
    '  /** Single precision FOUR. */\n' +
    '  #define FOUR 4.0f\n' +
    '#endif\n' +
    '\n' +
    '/** DEBUG definition. Used for barrier synchronization after kernel in GPU code. */\n' +
    '#define DEBUG\n' +
    '\n' +
    '/** IGN definition. Used to flag ignition delay calculation. */\n' +
    '//#define IGN\n' +
    '\n' +
    '/** PRINT definition. Used to flag printing of output values. */\n' +
    '//#define PRINT\n' +
    '\n' +
    '/** Definition of problem type.\n' +
    ' * CONV is constant volume.\n' +
    ' * CONP is constant pressure.\n' +
    ' */\n' +
    '#define CONV\n\n')

    file.write('/** Number of species.\n')
    for sp in specs:
        file.write(' * {:} {:}\n'.format(specs.index(sp), sp.name))
    file.write(' */\n')
    file.write('#define NSP {:}\n'.format(nsp))

    file.write('/** Number of variables. NN = NSP + 1 (temperature). */\n')
    file.write('#define NN {:}\n'.format(num_eq))
    file.write('\n')

    file.write('/** Unsigned int typedef. */\n')
    file.write('typedef unsigned int uint;\n')
    file.write('/** Unsigned short int typedef. */\n')
    file.write('typedef unsigned short int usint;\n')
    file.write('\n')

    file.write('#ifdef __cplusplus\n')
    file.write('extern "C" {\n')
    file.write('#endif\n')
    file.write('void mole2mass (const Real*, Real*);\n')
    file.write('void mass2mole (const Real*, Real*);\n')
    file.write('#ifdef __cplusplus\n')
    file.write('}\n')
    file.write('#endif\n')
    file.write('\n')

    file.close()

    return


def create_rate_subs(lang, mech_name, therm_name=None, last_spec=None):
    """Create rate subroutines from mechanism.

    Parameters
    ----------
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    mech_name : str
        Reaction mechanism filename (e.g. 'mech.dat').
    therm_name : str, optional
        Thermodynamic database filename (e.g. 'therm.dat')
        or nothing if info in mechanism file.
    last_spec : str, optional
        If specified, the species to assign to the last index.
        Typically should be N2, Ar, He or another inert bath gas

    Returns
    -------
    None

    """

    lang = lang.lower()
    if lang not in utils.langs:
        print('Error: language needs to be one of: ')
        for l in utils.langs:
            print(l)
        sys.exit()

    if lang in ['fortran', 'matlab']:
        print('WARNING: Fortran and Matlab support incomplete.')

    # create output directory if none exists
    build_path = './out/'
    utils.create_dir(build_path)

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if mech_name.endswith(tuple(['.cti', '.xml'])):
        [elems, specs, reacs] = mech.read_mech_ct(mech_name)
    else:
        [elems, specs, reacs] = mech.read_mech(mech_name, therm_name)

    # Check to see if the last_spec is specified
    if last_spec is not None:
        # Find the index if possible
        isp = next((i for i, sp in enumerate(specs)
                   if sp.name.lower() == last_spec.lower().strip()),
                   None
                   )
        if isp is None:
            print('Warning: User specified last species {} '
                  'not found in mechanism.'
                  '  Attempting to find a default species.'.format(last_spec)
                  )
            last_spec = None
        else:
            last_spec = isp
    else:
        print('User specified last species not found or not specified.  '
              'Attempting to find a default species')
    if last_spec is None:
        wt = chem.get_elem_wt()
        #check for N2, Ar, He, etc.
        candidates = [('N2', wt['n'] * 2.), ('Ar', wt['ar']),
                        ('He', wt['he'])]
        for sp in candidates:
            match = next((isp for isp, spec in enumerate(specs)
                          if sp[0].lower() == spec.name.lower() and
                          sp[1] == spec.mw),
                            None)
            if match is not None:
                last_spec = match
                break
        if last_spec is not None:
            print('Default last species '
                  '{} found.'.format(specs[last_spec].name)
                  )
    if last_spec is None:
        print('Warning: Neither a user specified or default last species '
              'could be found. Proceeding using the last species in the '
              'base mechanism: {}'.format(specs[-1].name))
        last_spec = len(specs) - 1

    # ordering of species and reactions not changed.
    fwd_rxn_mapping = range(len(reacs))
    spec_maps = utils.get_species_mappings(len(specs), last_spec)
    fwd_spec_mapping, reverse_spec_mapping = spec_maps

    #pick up the last_spec and drop it at the end
    temp = specs[:]
    for i in range(len(specs)):
        specs[i] = temp[fwd_spec_mapping[i]]

    ## Now begin writing subroutines

    # print reaction rate subroutine
    rate.write_rxn_rates(build_path, lang, specs, reacs, fwd_rxn_mapping)

    # if third-body/pressure-dependent reactions,
    # print modification subroutine
    if next((r for r in reacs if (r.thd_body or r.pdep)), None):
        rate.write_rxn_pressure_mod(build_path, lang, specs, reacs,
                                    fwd_rxn_mapping
                                    )

    # write species rates subroutine
    rate.write_spec_rates(build_path, lang, specs, reacs,
                          fwd_spec_mapping, fwd_rxn_mapping
                          )

    # write chem_utils subroutines
    rate.write_chem_utils(build_path, lang, specs)

    # write derivative subroutines
    rate.write_derivs(build_path, lang, specs, reacs)

    # write mass-mole fraction conversion subroutine
    rate.write_mass_mole(build_path, lang, specs)

    write_header(build_path, lang, specs)
    write_main(build_path, lang, specs)

    return 0


if __name__ == "__main__":
    # command line arguments
    parser = ArgumentParser(description='Generates source code for species '
                                        'and reaction rates.'
                            )
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output source files.'
                        )
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='Input mechanism filename (e.g., mech.dat).'
                        )
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism.'
                        )
    parser.add_argument('-ls', '--last_species',
                        required=False,
                        type=str,
                        default=None,
                        help='The name of the species to set as the last in '
                             'the mechanism. If not specifed, defaults to '
                             'the first of N2, AR, and HE in the mechanism.'
                        )

    args = parser.parse_args()

    create_rate_subs(args.lang, args.input, args.thermo)
