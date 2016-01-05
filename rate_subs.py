"""Module for writing species/reaction rate subroutines."""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math

# Local imports
import chem_utilities as chem
import mech_interpret as mech
import utils

__all__ = ['rxn_rate_const', 'write_rxn_rates', 'write_rxn_pressure_mod',
           'write_spec_rates', 'write_chem_utils', 'write_derivs']

def rxn_rate_const(A, b, E):
    """Returns line with reaction rate calculation (after = sign).

    Parameters
    ----------
    A : float
        Arrhenius pre-exponential coefficient
    b : float
        Arrhenius temperature exponent
    E : float
        Arrhenius activation energy

    Returns
    -------
    line : str
        String with expression for reaction rate.

    Notes
    -----
    Form of the reaction rate constant (from, e.g., Lu and Law [1]_):
    .. math::
        :nowrap:
        k_f = \begin{cases}
        A & \text{if } \beta = 0 \text{ and } T_a = 0 \\
        \exp \left( \log A + \beta \log T \right) &
        \text{if } \beta \neq 0 \text{ and } \text{if } T_a = 0 \\
        \exp \left( \log A + \beta \log T - T_a / T \right)	&
        \text{if } \beta \neq 0 \text{ and } T_a \neq 0 \\
        \exp \left( \log A - T_a / T \right)
        & \text{if } \beta = 0 \text{ and } T_a \neq 0 \\
        A \prod^b T	& \text{if } T_a = 0 \text{ and }
        b \in \mathbb{Z} \text{ (integers) }
        \end{cases}

    .. [1] TF Lu and CK Law, "Toward accommodating realistic fuel chemistry
       in large-scale computations," Progress in Energy and Combustion
       Science, vol. 35, pp. 192-215, 2009. doi:10.1016/j.pecs.2008.10.002
    """

    line = ''

    if A > 0:
        logA = math.log(A)

        if not E:
            # E = 0
            if not b:
                # b = 0
                line += str(A)
            else:
                # b != 0
                if isinstance(b, int):
                    line += str(A)
                    for i in range(b):
                        line += ' * T'
                else:
                    line += 'exp({:.16e}'.format(logA)
                    if b > 0:
                        line += ' + ' + str(b)
                    else:
                        line += ' - ' + str(abs(b))
                    line += ' * logT)'
        else:
            # E != 0
            if not b:
                # b = 0
                line += ('exp({:.16e}'.format(logA) +
                         ' - ({:.16e} / T))'.format(E)
                         )
            else:
                # b!= 0
                line += 'exp({:.16e}'.format(logA)
                if b > 0:
                    line += ' + ' + str(b)
                else:
                    line += ' - ' + str(abs(b))
                line += ' * logT - ({:.16e} / T))'.format(E)
    elif A < 0:
        #a < 0, can't take the log of it
        #the reaction, should also be a duplicate to make any sort of sense
        if not E:
            #E = 0
            if not b:
                #b = 0
                line += str(A)
            else:
                #b != 0
                if utils.is_integer(b):
                    line += str(A)
                    for i in range(int(b)):
                        line += ' * T'
                else:
                    line += '{:.16e} * exp('.format(A)
                    if b > 0:
                        line += str(b)
                    else:
                        line += '-' + str(abs(b))
                    line += ' * logT)'
        else:
            #E != 0
            if not b:
                # b = 0
                line += '{:.16e} * exp(-({:.16e} / T))'.format(A, E)
            else:
                # b!= 0
                line += '{:.16e} * exp('.format(A)
                if b > 0:
                    line += str(b)
                else:
                    line += '-' + str(abs(b))
                line += ' * logT - ({:.16e} / T))'.format(E)

    else:
      raise NotImplementedError

    return line


def get_cheb_rate(lang, rxn, write_defns=True):
    """Given a reaction, and a temperature and pressure, this routine
    will generate code to evaluate the Chebyshev rate efficiently.

    Assumes:
    Existence of variables dot_prod* of sized at least rxn.cheb_n_temp
    Pred and Tred, T and pres, and kf, cheb_temp_0 and cheb_temp_1
    """

    line_list = []
    tlim_inv_sum = 1.0 / rxn.cheb_tlim[0] + 1.0 / rxn.cheb_tlim[1]
    tlim_inv_sub = 1.0 / rxn.cheb_tlim[1] - 1.0 / rxn.cheb_tlim[0]
    if write_defns:
        line_list.append(
                'Tred = ((2.0 / T) - ' +
                '{:.8e}) / {:.8e}'.format(tlim_inv_sum, tlim_inv_sub)
                )

    plim_log_sum = (math.log10(rxn.cheb_plim[0]) +
                    math.log10(rxn.cheb_plim[1])
                    )
    plim_log_sub = (math.log10(rxn.cheb_plim[1]) -
                    math.log10(rxn.cheb_plim[0])
                    )
    if write_defns:
        line_list.append(
                'Pred = (2.0 * log10(pres) - ' +
                '{:.8e}) / {:.8e}'.format(plim_log_sum, plim_log_sub)
                )

    line_list.append('cheb_temp_0 = 1')
    line_list.append('cheb_temp_1 = Pred')
    #start pressure dot product
    for i in range(rxn.cheb_n_temp):
        line_list.append(utils.get_array(lang, 'dot_prod', i) +
          '= {:.8e} + Pred * {:.8e}'.format(rxn.cheb_par[i, 0],
            rxn.cheb_par[i, 1]))

    #finish pressure dot product
    update_one = True
    for j in range(2, rxn.cheb_n_pres):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = 'cheb_temp_{}'.format(old)
        line += ' = 2 * Pred * cheb_temp_{}'.format(new)
        line += ' - cheb_temp_{}'.format(old)
        line_list.append(line)
        for i in range(rxn.cheb_n_temp):
            line_list.append(utils.get_array(lang, 'dot_prod', i)  +
              ' += {:.8e} * cheb_temp_{}'.format(
                rxn.cheb_par[i, j], old))

        update_one = not update_one

    line_list.append('cheb_temp_0 = 1')
    line_list.append('cheb_temp_1 = Tred')
    #finally, do the temperature portion
    line_list.append('kf = ' + utils.get_array(lang, 'dot_prod', 0) +
                     ' + Tred * ' + utils.get_array(lang, 'dot_prod', 1))

    update_one = True
    for i in range(2, rxn.cheb_n_temp):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = 'cheb_temp_{}'.format(old)
        line += ' = 2 * Tred * cheb_temp_{}'.format(new)
        line += ' - cheb_temp_{}'.format(old)
        line_list.append(line)
        line_list.append('kf += ' + utils.get_array(lang, 'dot_prod', i) +
                         ' * ' + 'cheb_temp_{}'.format(old))

        update_one = not update_one

    line_list.append('kf = ' + utils.exp_10_fun[lang] + 'kf)')
    line_list = [utils.line_start + line + utils.line_end[lang] for
                  line in line_list]

    return ''.join(line_list)


def write_rxn_rates(path, lang, specs, reacs, fwd_rxn_mapping):
    """Write reaction rate subroutine.

    Includes conditionals for reversible reactions.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in the mechanism.
    reacs : list of ReacInfo
        List of reactions in the mechanism.
    fwd_rxn_mapping : List of integers
        The index of the reaction in the original mechanism

    Returns
    _______
    None

    """

    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
    num_rev = len(rev_reacs)
    pdep_reacs = [i for i, rxn in enumerate(reacs) if rxn.thd_body or rxn.pdep]

    pre  = '__device__ ' if lang == 'cuda' else ''
    filename = 'rates' + utils.header_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:
        file.write(
            '#ifndef RATES_HEAD\n'
            '#define RATES_HEAD\n'
            '\n'
            '#include "header{}"\n'.format(utils.header_ext[lang]) +
            '\n'
            '{0}void eval_rxn_rates (const double,'
            ' const double, const double*, double*, double*);\n'
            '{0}void eval_spec_rates (const double*,'
            ' const double*, const double*, double*, double*);\n'.format(pre)
            )

        if pdep_reacs:
            file.write('{}void get_rxn_pres_mod (const double, const '
                       'double, const double*, double*);\n'.format(pre)
                       )

        file.write('\n'
                   '#endif\n'
                   )

    filename = 'rxn_rates' + utils.file_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:
        line = ''
        if lang == 'cuda': line = '__device__ '

        if lang in ['c', 'cuda']:
            file.write('#include "rates' + utils.header_ext[lang] + '"\n')
            line += ('void eval_rxn_rates (const double T, const double pres,'
                     ' const double * C, double * fwd_rxn_rates, '
                     'double * rev_rxn_rates) {\n'
                     )
        elif lang == 'fortran':
            line += ('subroutine eval_rxn_rates(T, pres, C, fwd_rxn_rates,'
                     ' rev_rxn_rates)\n\n'
                     )

            # fortran needs type declarations
            line += ('  implicit none\n'
                     '  double precision, intent(in) :: '
                     'T, pres, C({})\n'.format(num_s)
                     )
            line += ('  double precision, intent(out) :: '
                     'fwd_rxn_rates({}), '.format(num_r) +
                     'rev_rxn_rates({})\n'.format(num_rev)
                     )
            line += ('  \n'
                     '  double precision :: logT\n'
                     )

            kf_flag = True
            if rev_reacs and any([not r.rev_par for r in reacs]):
                line += '  double precision :: kf, Kc\n'
                kf_flag = False

            if any([rxn.cheb for rxn in reacs]):
                if kf_flag:
                    line += '  double precision :: kf, Tred, Pred\n'
                    kf_flag = False
                else:
                    line += '  double precision :: Tred, Pred\n'
            if any([rxn.plog for rxn in reacs]):
                if kf_flag:
                    line += '  double precision :: kf, kf2\n'
                    kf_flag = False
                else:
                    line += '  double precision :: kf2\n'
            line += '\n'
        elif lang == 'matlab':
            line += ('function [fwd_rxn_rates, rev_rxn_rates] = '
                     'eval_rxn_rates (T, pres, C)\n\n'
                     '  fwd_rxn_rates = zeros({},1);\n'.format(num_r) +
                     '  rev_rxn_rates = fwd_rxn_rates;\n'
                     )
        file.write(line)

        get_array = utils.get_array

        pre = '  '
        if lang == 'c':
            pre += 'double '
        elif lang == 'cuda':
            pre += 'register double '
        line = (pre + 'logT = log(T)' +
                utils.line_end[lang]
                )
        file.write(line)
        file.write('\n')

        kf_flag = True
        if rev_reacs and any([not r.rev_par for r in reacs]):
            kf_flag = False

            if lang == 'c':
                file.write('  double kf;\n'
                           '  double Kc;\n'
                           )
            elif lang == 'cuda':
                file.write('  register double kf;\n'
                           '  register double Kc;\n'
                           )

        if any([rxn.cheb for rxn in reacs]):
            # Other variables needed for Chebyshev
            if lang == 'c':
                if kf_flag:
                    file.write('  double kf;\n')
                    kf_flag = False
                file.write('  double Tred;\n'
                           '  double Pred;\n')
                file.write(utils.line_start +
                           'double cheb_temp_0, cheb_temp_1' +
                           utils.line_end[lang]
                           )
                dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
                file.write(utils.line_start +
                           'double dot_prod[{}]'.format(dim) +
                           utils.line_end[lang]
                           )


            elif lang == 'cuda':
                if kf_flag:
                    file.write('  register double kf;\n')
                    kf_flag = False
                file.write('  register double Tred;\n'
                           '  register double Pred;\n')
                file.write(utils.line_start +
                           'double cheb_temp_0, cheb_temp_1' +
                           utils.line_end[lang]
                           )
                dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
                file.write(utils.line_start +
                           'double dot_prod[{}]'.format(dim) +
                           utils.line_end[lang]
                           )


        if any([rxn.plog for rxn in reacs]):
            # Variables needed for Plog
            if lang == 'c':
                if kf_flag:
                    file.write('  double kf;\n')
                file.write('  double kf2;\n')
            if lang == 'cuda':
                if kf_flag:
                    file.write('  register double kf;\n')
                file.write('  register double kf2;\n')

        file.write('\n')

        def __get_arrays(sp, factor=1.0):
            # put together all our coeffs
            lo_array = [nu * factor] + [
                sp.lo[6], sp.lo[0], sp.lo[0] - 1.0, sp.lo[1] / 2.0,
                sp.lo[2] / 6.0, sp.lo[3] / 12.0, sp.lo[4] / 20.0,
                sp.lo[5]
                ]

            lo_array = [x * lo_array[0] for x in
                        [lo_array[1] - lo_array[2]] + lo_array[3:]
                        ]

            hi_array = [nu * factor] + [
                sp.hi[6], sp.hi[0], sp.hi[0] - 1.0, sp.hi[1] / 2.0,
                sp.hi[2] / 6.0, sp.hi[3] / 12.0, sp.hi[4] / 20.0,
                sp.hi[5]
                ]

            hi_array = [x * hi_array[0] for x in
                        [hi_array[1] - hi_array[2]] + hi_array[3:]
                        ]
            return lo_array, hi_array

        for i_rxn in range(len(reacs)):
            file.write(utils.line_start + utils.comment[lang] +
                        'rxn {}'.format(fwd_rxn_mapping[i_rxn]) + '\n')
            rxn = reacs[i_rxn]

            # if reversible, save forward rate constant for use
            if rxn.rev and not rxn.rev_par and not (rxn.cheb or rxn.plog):
                line = ('  kf = ' + rxn_rate_const(rxn.A, rxn.b, rxn.E) +
                        utils.line_end[lang]
                        )
                file.write(line)
            elif rxn.cheb:
                file.write(get_cheb_rate(lang, rxn))
            elif rxn.plog:
                # Special forward rate evaluation for Plog reacions
                vals = rxn.plog_par[0]
                file.write('  if (pres <= {:.4e}) {{\n'.format(vals[0]) +
                           '    kf = ' +
                           rxn_rate_const(vals[1], vals[2], vals[3]) +
                           utils.line_end[lang]
                           )

                for idx, vals in enumerate(rxn.plog_par[:-1]):
                    vals2 = rxn.plog_par[idx + 1]

                    line = ('  }} else if ((pres > {:.4e}) '.format(vals[0]) +
                            '&& (pres <= {:.4e})) {{\n'.format(vals2[0]))
                    file.write(line)

                    line = ('    kf = log(' +
                            rxn_rate_const(vals[1], vals[2], vals[3]) + ')'
                            )
                    file.write(line + utils.line_end[lang])
                    line = ('    kf2 = log(' +
                            rxn_rate_const(vals2[1], vals2[2], vals2[3]) + ')'
                            )
                    file.write(line + utils.line_end[lang])

                    pres_log_diff = math.log(vals2[0]) - math.log(vals[0])
                    line = ('    kf = exp(kf + (kf2 - kf) * (log(pres) - ' +
                            '{:.16e}) / '.format(math.log(vals[0])) +
                            '{:.16e})'.format(pres_log_diff)
                            )
                    file.write(line + utils.line_end[lang])

                vals = rxn.plog_par[-1]
                file.write(
                        '  }} else if (pres > {:.4e}) {{\n'.format(vals[0]) +
                        '    kf = ' +
                        rxn_rate_const(vals[1], vals[2], vals[3]) +
                        utils.line_end[lang] +
                        '  }\n'
                        )

            line = '  ' + get_array(lang, 'fwd_rxn_rates', i_rxn) + ' = '

            # reactants
            for i, isp in enumerate(rxn.reac):
                nu = rxn.reac_nu[i]

                # check if stoichiometric coefficient is double or integer
                if utils.is_integer(nu):
                    # integer, so just use multiplication
                    for i in range(int(nu)):
                        line += '' + get_array(lang, 'C', isp) + ' * '
                else:
                    line += ('pow(' + get_array(lang, 'C', isp) +
                             ', {}) *'.format(nu)
                             )

            # Rate constant: print if not reversible, or reversible but
            # with explicit reverse parameters.
            if (rxn.rev and not rxn.rev_par) or rxn.plog or rxn.cheb:
                line += 'kf'
            else:
                line += rxn_rate_const(rxn.A, rxn.b, rxn.E)

            line += utils.line_end[lang]
            file.write(line)

            if rxn.rev:

                if not rxn.rev_par:

                    # line = '  Kc = 0.0' + utils.line_end[lang]
                    # file.write(line)

                    # sum of stoichiometric coefficients
                    sum_nu = 0

                    coeffs = {}
                    # go through product species
                    for isp, prod_sp in enumerate(rxn.prod):
                        # check if species also in reactants
                        if prod_sp in rxn.reac:
                            isp2 = rxn.reac.index(prod_sp)
                            nu = rxn.prod_nu[isp] - rxn.reac_nu[isp2]
                        else:
                            nu = rxn.prod_nu[isp]

                        # Skip species with zero overall
                        # stoichiometric coefficient.
                        if (nu == 0):
                            continue

                        sum_nu += nu

                        # get species object
                        sp = specs[prod_sp]
                        if not sp:
                            print('Error: species ' + prod_sp + ' in reaction '
                                  '{} not found.\n'.format(i_rxn)
                                  )
                            sys.exit()

                        lo_array, hi_array = __get_arrays(sp)

                        if not sp.Trange[1] in coeffs:
                            coeffs[sp.Trange[1]] = lo_array, hi_array
                        else:
                            coeffs[sp.Trange[1]] = [
                                lo_array[i] + coeffs[sp.Trange[1]][0][i]
                                for i in range(len(lo_array))
                                ], [
                                hi_array[i] + coeffs[sp.Trange[1]][1][i]
                                for i in range(len(hi_array))
                                ]

                    # now loop through reactants
                    for isp, reac_sp in enumerate(rxn.reac):
                        # Check if species also in products;
                        # if so, already considered).
                        if reac_sp in rxn.prod: continue

                        nu = rxn.reac_nu[isp]
                        sum_nu -= nu

                        # get species object
                        sp = specs[reac_sp]
                        if not sp:
                            print('Error: species ' + reac_sp + ' in reaction '
                                  '{} not found.\n'.format(i_rxn)
                                  )
                            sys.exit()

                        lo_array, hi_array = __get_arrays(sp, factor=-1.0)

                        if not sp.Trange[1] in coeffs:
                            coeffs[sp.Trange[1]] = lo_array, hi_array
                        else:
                            coeffs[sp.Trange[1]] = [
                                lo_array[i] +
                                coeffs[sp.Trange[1]][0][i]
                                for i in range(len(lo_array))
                                ], [hi_array[i] +
                                coeffs[sp.Trange[1]][1][i]
                                for i in range(len(hi_array))
                                ]

                    isFirst = True
                    for T_mid in coeffs:
                        # need temperature conditional for equilibrium constants
                        line = '  if (T <= {:})'.format(T_mid)
                        if lang in ['c', 'cuda']:
                            line += ' {\n'
                        elif lang == 'fortran':
                            line += ' then\n'
                        elif lang == 'matlab':
                            line += '\n'
                        file.write(line)

                        lo_array, hi_array = coeffs[T_mid]

                        if isFirst:
                            line = '    Kc = '
                        else:
                            if lang in ['cuda', 'c']:
                                line = '    Kc += '
                            else:
                                line = '    Kc = Kc + '
                        line += ('({:.16e} + '.format(lo_array[0]) +
                                 '{:.16e} * '.format(lo_array[1]) +
                                 'logT + T * ('
                                 '{:.16e} + T * ('.format(lo_array[2]) +
                                 '{:.16e} + T * ('.format(lo_array[3]) +
                                 '{:.16e} + '.format(lo_array[4]) +
                                 '{:.16e} * T))) - '.format(lo_array[5]) +
                                 '{:.16e} / T)'.format(lo_array[6]) +
                                 utils.line_end[lang]
                                 )
                        file.write(line)

                        if lang in ['c', 'cuda']:
                            file.write('  } else {\n')
                        elif lang in ['fortran', 'matlab']:
                            file.write('  else\n')

                        if isFirst:
                            line = '    Kc = '
                        else:
                            if lang in ['cuda', 'c']:
                                line = '    Kc += '
                            else:
                                line = '    Kc = Kc + '
                        line += ('({:.16e} + '.format(hi_array[0]) +
                                 '{:.16e} * '.format(hi_array[1]) +
                                 'logT + T * ('
                                 '{:.16e} + T * ('.format(hi_array[2]) +
                                 '{:.16e} + T * ('.format(hi_array[3]) +
                                 '{:.16e} + '.format(hi_array[4]) +
                                 '{:.16e} * T))) - '.format(hi_array[5]) +
                                 '{:.16e} / T)'.format(hi_array[6]) +
                                 utils.line_end[lang]
                                 )
                        file.write(line)

                        if lang in ['c', 'cuda']:
                            file.write('  }\n\n')
                        elif lang == 'fortran':
                            file.write('  end if\n\n')
                        elif lang == 'matlab':
                            file.write('  end\n\n')
                        isFirst = False

                    line = ('  Kc = '
                            '{:.16e}'.format((chem.PA / chem.RU) ** sum_nu) +
                            ' * exp(Kc)' +
                            utils.line_end[lang]
                            )
                    file.write(line)

                line = '  ' + get_array(lang, 'rev_rxn_rates',
                                        rev_reacs.index(i_rxn)
                                        ) + ' = '

                # reactants (products from forward reaction)
                for isp in rxn.prod:
                    nu = rxn.prod_nu[rxn.prod.index(isp)]

                    # check if stoichiometric coefficient is double or integer
                    if utils.is_integer(nu):
                        # integer, so just use multiplication
                        for i in range(int(nu)):
                            line += '' + get_array(lang, 'C', isp) + ' * '
                    else:
                        line += ('pow(' + get_array(lang, 'C', isp) +
                                 ', {}) * '.format(nu)
                                 )

                # rate constant
                if rxn.rev_par:
                    # explicit reverse Arrhenius parameters
                    line += rxn_rate_const(rxn.rev_par[0],
                                           rxn.rev_par[1],
                                           rxn.rev_par[2]
                                           )
                else:
                    # use equilibrium constant
                    line += 'kf / Kc'
                line += utils.line_end[lang]
                file.write(line)

                file.write('\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_rxn_rates\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_rxn_rates\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

    return


def write_rxn_pressure_mod(path, lang, specs, reacs, fwd_rxn_mapping):
    """Write subroutine to for reaction pressure dependence modifications.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    specs : list of SpecInfo
        List of species in mechanism.
    reacs : list of ReacInfo
        List of reactions in mechanism.
    fwd_rxn_mapping : List of integers
        The order of the reaction in the original mechanism

    Returns
    -------
    None

    """
    filename = 'rxn_rates_pres_mod' + utils.file_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:

        # headers
        if lang in ['c', 'cuda']:
            file.write('#include <math.h>\n'
                       '#include "header{}"\n'.format(utils.header_ext[lang])
                       )
            file.write('\n')

        # list of reactions with third-body or pressure-dependence
        pdep_reacs = []
        thd_flag = False
        pdep_flag = False
        troe_flag = False
        sri_flag = False
        for i_rxn, reac in enumerate(reacs):
            if reac.thd_body:
                # add reaction index to list
                thd_flag = True
                pdep_reacs.append(i_rxn)
            elif reac.pdep:
                # add reaction index to list
                pdep_flag = True
                pdep_reacs.append(i_rxn)

                if reac.troe and not troe_flag: troe_flag = True
                if reac.sri and not sri_flag: sri_flag = True

        line = ''
        if lang == 'cuda': line = '__device__ '

        if lang in ['c', 'cuda']:
            line += (
                'void get_rxn_pres_mod (const double T, const double pres, '
                'const double * C, double * pres_mod) {\n'
                )
        elif lang == 'fortran':
            line += 'subroutine get_rxn_pres_mod ( T, pres, C, pres_mod )\n\n'

            # fortran needs type declarations
            line += ('  implicit none\n'
                     '  double precision, intent(in) :: T, pres, '
                     'C({})\n'.format(len(specs)) +
                     '  double precision, intent(out) :: '
                     'pres_mod({})\n'.format(len(pdep_reacs)) +
                     '  \n'
                     '  double precision :: logT, m\n')
        elif lang == 'matlab':
            line += ('function pres_mod = get_rxn_pres_mod (T, pres, C)\n\n'
                     '  pres_mod = zeros({},1);\n'.format(len(pdep_reacs))
                     )
        file.write(line)

        get_array = utils.get_array

        # declarations for third-body variables
        if thd_flag or pdep_flag:
            if lang == 'c':
                file.write('  // third body variable declaration\n'
                           '  double thd;\n'
                           '\n'
                           )
            elif lang == 'cuda':
                file.write('  // third body variable declaration\n'
                           '  register double thd;\n'
                           '\n'
                           )
            elif lang == 'fortran':
                file.write('  ! third body variable declaration\n'
                           '  double precision :: thd\n'
                           )

        # declarations for pressure-dependence variables
        if pdep_flag:
            if lang == 'c':
                file.write('  // pressure dependence variable declarations\n'
                           '  double k0;\n'
                           '  double kinf;\n'
                           '  double Pr;\n'
                           '\n'
                           )
                if troe_flag:
                    # troe variables
                    file.write('  // troe variable declarations\n'
                               '  double logFcent;\n'
                               '  double A;\n'
                               '  double B;\n'
                               '\n'
                               )
                if sri_flag:
                    # sri variables
                    file.write('  // sri variable declarations\n')
                    file.write('  double X;\n'
                               '\n'
                               )
            elif lang == 'cuda':
                file.write('  // pressure dependence variable declarations\n')
                file.write('  register double k0;\n'
                           '  register double kinf;\n'
                           '  register double Pr;\n'
                           '\n'
                           )
                if troe_flag:
                    # troe variables
                    file.write('  // troe variable declarations\n'
                               '  register double logFcent;\n'
                               '  register double A;\n'
                               '  register double B;\n'
                               '\n'
                               )
                if sri_flag:
                    # sri variables
                    file.write('  // sri variable declarations\n')
                    file.write('  register double X;\n'
                               '\n')
            elif lang == 'fortran':
                file.write('  ! pressure dependence variable declarations\n'
                           '  double precision :: k0, kinf, Pr\n'
                           '\n'
                           )
                if troe_flag:
                    # troe variables
                    file.write('  ! troe variable declarations\n'
                               '  double precision :: logFcent, A, B\n'
                               '\n'
                               )
                if sri_flag:
                    # sri variables
                    file.write('  ! sri variable declarations\n')
                    file.write('  double precision :: X\n'
                               '\n')

        if lang == 'c':
            file.write('  double logT = log(T);\n'
                       '  double m = pres / ({:.8e} * T);\n'.format(chem.RU)
                       )
        elif lang == 'cuda':
            file.write('  register double logT = log(T);\n'
                       '  register double m = pres / ('
                       '{:.8e} * T);\n'.format(chem.RU)
                       )
        elif lang == 'fortran':
            file.write('  logT = log(T)\n'
                       '  m = pres / ({:.8e} * T)\n'.format(chem.RU)
                       )
        elif lang == 'matlab':
            file.write('  logT = log(T);\n'
                       '  m = pres / ({:.8e} * T);\n'.format(chem.RU)
                       )

        file.write('\n')

        pind = 0
        # loop through third-body and pressure-dependent reactions
        for rind in range(len(reacs)):
            reac = reacs[rind]  # index in reaction list

            if not (reac.pdep or reac.thd_body):
                continue

            printind = fwd_rxn_mapping[rind]
            # print reaction index
            if lang in ['c', 'cuda']:
                line = '  // reaction ' + str(printind)
            elif lang == 'fortran':
                line = '  ! reaction ' + str(printind + 1)
            elif lang == 'matlab':
                line = '  % reaction ' + str(printind + 1)
            line += utils.line_end[lang]
            file.write(line)

            # third-body reaction
            if reac.thd_body:

                line = '  ' + get_array(lang, 'pres_mod', pind) + ' = m'

                for sp in reac.thd_body_eff:
                    if sp[1] == 1.0:
                        continue
                    elif sp[1] > 1.0:
                        line += ' + {}'.format(sp[1] - 1.0)
                    elif sp[1] < 1.0:
                        line += ' - {}'.format(1.0 - sp[1])
                    line += ' * ' + get_array(lang, 'C', sp[0])

                line += utils.line_end[lang]
                file.write(line)

            # pressure dependence
            if reac.pdep:
                if reac.pdep_sp == '':
                    line = '  thd = m'
                    for sp in reac.thd_body_eff:
                        if sp[1] == 1.0:
                            continue
                        elif sp[1] > 1.0:
                            line += ' + {}'.format(sp[1] - 1.0)
                        elif sp[1] < 1.0:
                            line += ' - {}'.format(1.0 - sp[1])
                        line += ' * ' + get_array(lang, 'C', sp[0])
                    file.write(line + utils.line_end[lang])

                # low-pressure limit rate
                line = '  k0 = '
                if reac.low:
                    line += rxn_rate_const(reac.low[0],
                                           reac.low[1],
                                           reac.low[2]
                                           )
                else:
                    line += rxn_rate_const(reac.A, reac.b, reac.E)

                line += utils.line_end[lang]
                file.write(line)

                # high-pressure limit rate
                line = '  kinf = '
                if reac.high:
                    line += rxn_rate_const(reac.high[0],
                                           reac.high[1],
                                           reac.high[2]
                                           )
                else:
                    line += rxn_rate_const(reac.A, reac.b, reac.E)

                line += utils.line_end[lang]
                file.write(line)

                # reduced pressure
                if reac.pdep_sp != '':
                    line = ('  Pr = k0 * ' +
                            get_array(lang, 'C', reac.pdep_sp) + ' / kinf'
                            )
                else:
                    line = '  Pr = k0 * thd / kinf'
                line += utils.line_end[lang]
                file.write(line)

                simple = False
                if reac.troe:
                    # Troe form
                    line = ('  logFcent = log10( fmax('
                            '{:.8e} * '.format(1.0 - reac.troe_par[0])
                            )
                    if reac.troe_par[1] > 0.0:
                        line += 'exp(-T / {:.8e})'.format(reac.troe_par[1])
                    else:
                        line += 'exp(T / {:.8e})'.format(abs(reac.troe_par[1]))

                    line += ' + {:.8e} * '.format(reac.troe_par[0])
                    if reac.troe_par[2] > 0.0:
                        line += 'exp(-T / {:.8e})'.format(reac.troe_par[2])
                    else:
                        line += 'exp(T / {:.8e})'.format(abs(reac.troe_par[2]))

                    if len(reac.troe_par) == 4 and reac.troe_par[3] != 0.0:
                        line += ' + '
                        if reac.troe_par[3] > 0.0:
                            line += 'exp(-{:.8e} / T)'.format(reac.troe_par[3])
                        else:
                            line += 'exp({:.8e} / T)'.format(abs(reac.troe_par[3]))
                    line += ', 1.0e-300))' + utils.line_end[lang]
                    file.write(line)

                    line = ('  A = log10(fmax(Pr, 1.0e-300)) - '
                            '0.67 * logFcent - 0.4' +
                            utils.line_end[lang]
                            )
                    file.write(line)

                    line = ('  B = 0.806 - 1.1762 * logFcent - '
                            '0.14 * log10(fmax(Pr, 1.0e-300))' +
                            utils.line_end[lang]
                            )
                    file.write(line)

                    line = ('  ' + get_array(lang, 'pres_mod', pind) +
                            ' = ' + utils.exp_10_fun[lang]
                            )
                    line += 'logFcent / (1.0 + A * A / (B * B))) '

                elif reac.sri:
                    # SRI form

                    line = ('  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * '
                            'log10(fmax(Pr, 1.0e-300)))' +
                            utils.line_end[lang]
                            )
                    file.write(line)

                    line = '  ' + get_array(lang, 'pres_mod', pind)
                    line += ' = pow({:.6} * '.format(reac.sri_par[0])
                    # Need to check for negative parameters, and
                    # skip "-" sign if so.
                    if reac.sri_par[1] > 0.0:
                        line += 'exp(-{:.6} / T)'.format(reac.sri_par[1])
                    else:
                        line += 'exp({:.6} / T)'.format(abs(reac.sri_par[1]))

                    if reac.sri_par[2] > 0.0:
                        line += ' + exp(-T / {:.6}), X) '.format(reac.sri_par[2])
                    else:
                        line += ' + exp(T / {:.6}), X) '.format(abs(reac.sri_par[2]))

                    if (len(reac.sri_par) == 5 and
                            reac.sri_par[3] != 1.0 and reac.sri_par[4] != 0.0):
                        line += ('* {:.8e} * '.format(reac.sri_par[3]) +
                                 'pow(T, {:.6}) '.format(reac.sri_par[4])
                                 )
                else:
                    # simple falloff fn (i.e. F = 1)
                    simple = True
                    line = '  ' + get_array(lang, 'pres_mod', pind) + ' = '
                    # regardless of F formulation
                    if reac.low:
                        # unimolecular/recombination fall-off reaction
                        line += ' Pr / (1.0 + Pr)'
                    elif reac.high:
                        # chemically-activated bimolecular reaction
                        line += '1.0 / (1.0 + Pr)'

                if not simple:
                    # regardless of F formulation
                    if reac.low:
                        # unimolecular/recombination fall-off reaction
                        line += '* Pr / (1.0 + Pr)'
                    elif reac.high:
                        # chemically-activated bimolecular reaction
                        line += '/ (1.0 + Pr)'

                line += utils.line_end[lang]
                file.write(line)

            # space in between each reaction
            file.write('\n')
            pind += 1

        if lang in ['c', 'cuda']:
            file.write('} // end get_rxn_pres_mod\n\n')
        elif lang == 'fortran':
            file.write('end subroutine get_rxn_pres_mod\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

    return


def write_spec_rates(path, lang, specs, reacs, fwd_spec_mapping, fwd_rxn_mapping):
    """Write subroutine to evaluate species rates of production.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in mechanism.
    reacs : list of ReacInfo
        List of reactions in mechanism.
    fwd_spec_mapping : list of int
        the index of the species in the original mechanism
    fwd_rxn_mapping : list of int
        the index of the reactions in the original mechanism

    Returns
    -------
    None

    """

    filename = 'spec_rates' + utils.file_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:

        if lang in ['c', 'cuda']:
            file.write('#include "header{}"\n'.format(utils.header_ext[lang])
                       )
            file.write('\n')

        num_s = len(specs)
        num_r = len(reacs)
        rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
        num_rev = len(rev_reacs)

        # pressure dependent reactions
        pdep_reacs = []
        for i_rxn, reac in enumerate(reacs):
            if reac.thd_body or reac.pdep:
                # add reaction index to list
                pdep_reacs.append(i_rxn)

        line = ''
        if lang == 'cuda': line = '__device__ '

        if lang in ['c', 'cuda']:
            line += ('void eval_spec_rates (const double * fwd_rates,'
                     ' const double * rev_rates, const double * pres_mod,'
                     ' double * sp_rates, double * dy_N) {\n'
                     )
        elif lang == 'fortran':
            line += ('subroutine eval_spec_rates (fwd_rates, rev_rates,'
                     ' pres_mod, sp_rates, dy_N)\n\n'
                     )

            # fortran needs type declarations
            line += '  implicit none\n'
            line += ('  double precision, intent(in) :: '
                     'fwd_rates({0}), rev_rates({0}), '.format(num_r) +
                     'pres_mod({})\n'.format(len(pdep_reacs))
                     )
            line += ('  double precision, intent(out) :: '
                     'sp_rates({}), dy_N\n'.format(num_s) +
                     '\n'
                     )
        elif lang == 'matlab':
            line += ('function sp_rates = eval_spec_rates (fwd_rates,'
                     ' rev_rates, pres_mod, dy_N)\n\n'
                     )
            line += '  sp_rates = zeros({},1);\n'.format(len(specs))
        file.write(line)

        get_array = utils.get_array

        def __get_var(spind):
            if spind + 1 == len(specs):
                line = '  ' + get_array(lang, '(*dy_N)', None)
            else:
                line = '  ' + get_array(lang, 'sp_rates', spind)
            return line
        new_loads = []
        cuda_loaded = [False for spec in specs]
        seen = [False for spec in specs]
        #loop through reaction
        for rind in range(len(reacs)):
            print_ind = fwd_rxn_mapping[rind]
            file.write(utils.line_start + utils.comment[lang] +
                        'rxn {}'.format(print_ind) + '\n')
            rxn = reacs[rind]
            #get allowed species
            my_specs = set(rxn.reac + rxn.prod)


            # loop through species
            for spind in my_specs:
                sp = specs[spind]

                #find nu
                nu = utils.get_nu(spind, rxn)
                if nu == 0.0:
                    continue

                file.write(utils.line_start + utils.comment[lang] +
                    'sp {}'.format(fwd_spec_mapping[spind]) + '\n')

                sign = '-' if nu < 0 else '+'
                line = __get_var(spind)
                line += ' {}= '.format(sign if seen[spind] else '')
                if not seen[spind] and nu < 0:
                    line += sign
                nu = abs(nu)
                if nu != 1.0:
                    if utils.is_integer(nu):
                        line += '{} * '.format(float(nu))
                    else:
                        line += '{:3} * '.format(nu)
                if rxn.rev:
                    rxn_out = (
                        '(' + get_array(lang, 'fwd_rates', rind) +
                        ' - ' + get_array(lang, 'rev_rates',
                                    rev_reacs.index(rind)) + ')'
                        )
                else:
                    rxn_out = get_array(lang, 'fwd_rates', rind)

                seen[spind] = True

                # pressure dependence modification
                if rxn.thd_body or rxn.pdep:
                    pind = pdep_reacs.index(rind)
                    rxn_out += ' * ' + get_array(lang, 'pres_mod', pind)

                # if lang == 'cuda':
                #    rxn_out = get_array(lang, rxn_out, None, preformed=True)
                line += rxn_out

                # done with this species
                line += utils.line_end[lang]
                file.write(line)
            file.write('\n')


        for i, seen_sp in enumerate(seen):
            if not seen_sp:
                file.write(utils.line_start + utils.comment[lang] +
                    'sp {}'.format(fwd_spec_mapping[i]) + '\n')
                file.write(__get_var(i) +
                           ' = 0.0' + utils.line_end[lang]
                           )

        if lang in ['c', 'cuda']:
            file.write('} // end eval_spec_rates\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_spec_rates\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

    return


def write_chem_utils(path, lang, specs):
    """Write subroutine to evaluate species thermodynamic properties.

    Notes
    -----
    Thermodynamic properties include:  enthalpy, energy, specific heat
    (constant pressure and volume).

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in the mechanism.

    Returns
    -------
    None

    """

    num_s = len(specs)
    pre = '__device__ ' if lang == 'cuda' else ''

    if lang in ['c', 'cuda']:
        filename = 'chem_utils' + utils.header_ext[lang]
        with open(os.path.join(path, filename), 'w') as file:
            file.write('#ifndef CHEM_UTILS_HEAD\n'
                       '#define CHEM_UTILS_HEAD\n'
                       '\n'
                       '#include "header{}"\n'.format(utils.header_ext[lang]) +
                       '\n'
                       '{0}void eval_conc (const double, const double, '
                       'const double*, double*, double*, double*, double*);\n'
                       '{0}void eval_conc_rho (const double, const double, '
                       'const double*, double*, double*, double*, double*);\n'
                       '{0}void eval_h (const double, double*);\n'
                       '{0}void eval_u (const double, double*);\n'
                       '{0}void eval_cv (const double, double*);\n'
                       '{0}void eval_cp (const double, double*);\n'
                       '\n'
                       '#endif\n'.format(pre)
                       )

    filename = 'chem_utils' + utils.file_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:

        if lang in ['c', 'cuda']:
            file.write('#include "header{}"\n'.format(utils.header_ext[lang]))
            file.write('\n')


        ###################################
        # species concentrations subroutine
        ###################################
        line = pre
        if lang in ['c', 'cuda']:
            line += ('void eval_conc (const double T, const double pres, '
                     'const double * y, double * y_N, double * mw_avg, '
                     'double * rho, double * conc) {\n\n'
                     )
        elif lang == 'fortran':
            line += (
                # fortran needs type declarations
                'subroutine eval_conc (T, pres, y, y_N, '
                'mw_avg, rho, conc)\n\n'
                '  implicit none\n'
                '  double precision, intent(in) :: T, pres, '
                'mass_frac\n'.format(num_s) +
                '  double precision, intent(out) :: y_N, mw_avg, '
                'rho, conc({})\n'.format(num_s) +
                '\n'
            )
        elif lang == 'matlab':
            line += ('function conc = eval_conc (T, pres, y, y_N, '
                     'mw_avg, rho, conc)\n\n'
                     )
        file.write(line)

        isfirst = True
        # Get mass fraction of last species
        file.write('  // mass fraction of final species\n')
        line = '  *y_N = 1.0 - ('
        for isp in range(len(specs[:-1])):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '               '

            if not isfirst: line += ' + '

            line += utils.get_array(lang, 'y', isp)

            isfirst = False
        line += ')'
        file.write(line + utils.line_end[lang])

        # calculation of mw avg
        line = '  *mw_avg = '
        isfirst = True
        for isp, sp in enumerate(specs[:-1]):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '     '

            if not isfirst: line += ' + '
            if lang in ['c', 'cuda']:
                line += ('(y[{}] * {:.16e})'.format(isp,
                         1.0 / sp.mw)
                         )
            elif lang in ['fortran', 'matlab']:
                line += ('(y[{}] * {:.16e})'.format(isp + 1,
                         1.0 / sp.mw)
                         )

            isfirst = False
        line += ' + ((*y_N) * {:.16e})'.format(1.0 / specs[-1].mw)
        line += utils.line_end[lang]
        file.write(line)
        file.write('  *mw_avg = 1.0 / *mw_avg;\n')

        # calculation of density
        file.write('  // mass-averaged density\n')
        line = '  *rho = pres * (*mw_avg) / ({:.8e} * T)'.format(chem.RU)
        file.write(line + utils.line_end[lang])

        # calculation of species molar concentrations

        # loop through species
        for isp, sp in enumerate(specs[:-1]):
            line = '  conc'
            if lang in ['c', 'cuda']:
                line += '[{0}] = (*rho) * y[{0}] * '.format(isp)
            elif lang in ['fortran', 'matlab']:
                line += '({0}) = (*rho) * y({0}) * '.format(isp + 1)
            line += '{:.16e}'.format(1.0 / sp.mw) + utils.line_end[lang]
            file.write(line)
        line = '  conc'
        if lang in ['c', 'cuda']:
            line += '[{0}] = (*rho) * (*y_N) * '.format(len(specs) - 1)
        elif lang in ['fortran', 'matlab']:
            line += '({0}) = (*rho) * y_N * '.format(len(specs))
        line += '{:.16e}'.format(1.0 / specs[-1].mw) + utils.line_end[lang]
        file.write(line + '\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_conc\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_conc\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

        #######################################################################

        line = pre
        if lang in ['c', 'cuda']:
            line += ('void eval_conc_rho (const double T, const double rho, '
                     'const double * y, double * y_N, double * mw_avg, '
                     'double * pres, double * conc) {\n\n'
                     )
        elif lang == 'fortran':
            line += (
                # fortran needs type declarations
                'subroutine eval_conc_rho (temp, rho, mass_frac, y_N, '
                'mw_avg, pres, conc)\n'
                '  implicit none\n'
                '  double precision, intent(in) :: '
                'T, rho, mass_frac({})\n'.format(num_s) +
                '  double precision, intent(out) :: '
                'conc({}), y_N, mw_avg, pres\n'.format(num_s) +
                '\n'
                )
        elif lang == 'matlab':
            line += ('function conc = eval_conc_rho (temp, rho, mass_frac, '
                     'y_N, mw_avg, pres, conc)\n\n'
                     )
        file.write(line)

        # Get mass fraction of last species
        file.write('  // mass fraction of final species\n')
        line = '  *y_N = 1.0 - ('
        isfirst = True
        for isp in range(len(specs[:-1])):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '               '

            if not isfirst: line += ' + '

            line += utils.get_array(lang, 'y', isp)

            isfirst = False
        line += ')'
        file.write(line + utils.line_end[lang])

        # calculation of mw avg
        line = '  *mw_avg = '
        isfirst = True
        for isp, sp in enumerate(specs[:-1]):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '     '

            if not isfirst: line += ' + '
            if lang in ['c', 'cuda']:
                line += ('(y[{}] * {:.16e})'.format(isp,
                         1.0 / sp.mw)
                         )
            elif lang in ['fortran', 'matlab']:
                line += ('(y[{}] * {})'.format(isp + 1,
                         1.0 / sp.mw)
                         )

            isfirst = False
        line += ' + ((*y_N) * {:.16e})'.format(1.0 / specs[-1].mw)
        line += utils.line_end[lang]
        file.write(line)
        file.write('  *mw_avg = 1.0 / *mw_avg;\n')

        # calculation of pressure
        file.write('  // pressure\n')
        line = '  *pres = rho * {:.8e} * T / (*mw_avg)'.format(chem.RU)
        file.write(line + utils.line_end[lang])

        # calculation of species molar concentrations

        # loop through species
        for isp, sp in enumerate(specs[:-1]):
            line = '  conc'
            if lang in ['c', 'cuda']:
                line += '[{0}] = rho * y[{0}] * '.format(isp)
            elif lang in ['fortran', 'matlab']:
                line += '({0}) = rho * y({0}) * '.format(isp + 1)
            line += '{:.16e}'.format(1.0 / sp.mw) + utils.line_end[lang]
            file.write(line)
        line = '  conc'
        if lang in ['c', 'cuda']:
            line += '[{0}] = rho * (*y_N) * '.format(len(specs) - 1)
        elif lang in ['fortran', 'matlab']:
            line += '({0}) = rho * y_N * '.format(len(specs))
        line += '{:.16e}'.format(1.0 / specs[-1].mw) + utils.line_end[lang]
        file.write(line)

        file.write('\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_conc\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_conc\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

        ######################
        # enthalpy subroutine
        ######################
        line = pre
        if lang in ['c', 'cuda']:
            line += 'void eval_h (const double T, double * h) {\n\n'
        elif lang == 'fortran':
            line += ('subroutine eval_h (T, h)\n\n'
                     # fortran needs type declarations
                     '  implicit none\n'
                     '  double precision, intent(in) :: T\n'
                     '  double precision, intent(out) :: h({})\n'.format(num_s) +
                     '\n'
                     )
        elif lang == 'matlab':
            line += 'function h = eval_h (T)\n\n'
        file.write(line)

        # loop through species
        for isp, sp in enumerate(specs):
            line = '  if (T <= {:})'.format(sp.Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)

            line = '    ' + utils.get_array(lang, 'h', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} + T * ('.format(sp.lo[5]) +
                     '{:.16e} + T * ('.format(sp.lo[0]) +
                     '{:.16e} + T * ('.format(sp.lo[1] / 2.0) +
                     '{:.16e} + T * ('.format(sp.lo[2] / 3.0) +
                     '{:.16e} + '.format(sp.lo[3] / 4.0) +
                     '{:.16e} * T)))))'.format(sp.lo[4] / 5.0) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')

            line = '    ' + utils.get_array(lang, 'h', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} + T * ('.format(sp.hi[5]) +
                     '{:.16e} + T * ('.format(sp.hi[0]) +
                     '{:.16e} + T * ('.format(sp.hi[1] / 2.0) +
                     '{:.16e} + T * ('.format(sp.hi[2] / 3.0) +
                     '{:.16e} + '.format(sp.hi[3] / 4.0) +
                     '{:.16e} * T)))))'.format(sp.hi[4] / 5.0) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  }\n\n')
            elif lang == 'fortran':
                file.write('  end if\n\n')
            elif lang == 'matlab':
                file.write('  end\n\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_h\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_h\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

        #################################
        # internal energy subroutine
        #################################
        line = pre
        if lang in ['c', 'cuda']:
            line += 'void eval_u (const double T, double * u) {\n\n'
        elif lang == 'fortran':
            line += ('subroutine eval_u (T, u)\n\n'
                     # fortran needs type declarations
                     '  implicit none\n'
                     '  double precision, intent(in) :: T\n'
                     '  double precision, intent(out) :: u({})\n'.format(num_s) +
                     '\n')
        elif lang == 'matlab':
            line += 'function u = eval_u (T)\n\n'
        file.write(line)

        # loop through species
        for isp, sp in enumerate(specs):
            line = '  if (T <= {:})'.format(sp.Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)

            line = '    ' + utils.get_array(lang, 'u', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} + T * ('.format(sp.lo[5]) +
                     '{:.16e} - 1.0 + T * ('.format(sp.lo[0]) +
                     '{:.16e} + T * ('.format(sp.lo[1] / 2.0) +
                     '{:.16e} + T * ('.format(sp.lo[2] / 3.0) +
                     '{:.16e} + '.format(sp.lo[3] / 4.0) +
                     '{:.16e} * T)))))'.format(sp.lo[4] / 5.0) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')

            line = '    ' + utils.get_array(lang, 'u', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} + T * ('.format(sp.hi[5]) +
                     '{:.16e} - 1.0 + T * ('.format(sp.hi[0]) +
                     '{:.16e} + T * ('.format(sp.hi[1] / 2.0) +
                     '{:.16e} + T * ('.format(sp.hi[2] / 3.0) +
                     '{:.16e} + '.format(sp.hi[3] / 4.0) +
                     '{:.16e} * T)))))'.format(sp.hi[4] / 5.0) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  }\n\n')
            elif lang == 'fortran':
                file.write('  end if\n\n')
            elif lang == 'matlab':
                file.write('  end\n\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_u\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_u\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

        ##################################
        # cv subroutine
        ##################################
        if lang in ['c', 'cuda']:
            line = pre + 'void eval_cv (const double T, double * cv) {\n\n'
        elif lang == 'fortran':
            line = ('subroutine eval_cv (T, cv)\n\n'
                    # fortran needs type declarations
                    '  implicit none\n'
                    '  double precision, intent(in) :: T\n'
                    '  double precision, intent(out) :: cv({})\n'.format(num_s) +
                    '\n'
                    )
        elif lang == 'matlab':
            line = 'function cv = eval_cv (T)\n\n'
        file.write(line)

        # loop through species
        for isp, sp in enumerate(specs):
            line = '  if (T <= {:})'.format(sp.Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)

            line = '    ' + utils.get_array(lang, 'cv', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} - 1.0 + T * ('.format(sp.lo[0]) +
                     '{:.16e} + T * ('.format(sp.lo[1]) +
                     '{:.16e} + T * ('.format(sp.lo[2]) +
                     '{:.16e} + '.format(sp.lo[3]) +
                     '{:.16e} * T))))'.format(sp.lo[4]) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')

            line = '    ' + utils.get_array(lang, 'cv', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} - 1.0 + T * ('.format(sp.hi[0]) +
                     '{:.16e} + T * ('.format(sp.hi[1]) +
                     '{:.16e} + T * ('.format(sp.hi[2]) +
                     '{:.16e} + '.format(sp.hi[3]) +
                     '{:.16e} * T))))'.format(sp.hi[4]) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  }\n\n')
            elif lang == 'fortran':
                file.write('  end if\n\n')
            elif lang == 'matlab':
                file.write('  end\n\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_cv\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_cv\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

        ###############################
        # cp subroutine
        ###############################
        if lang in ['c', 'cuda']:
            line = pre + 'void eval_cp (const double T, double * cp) {\n\n'
        elif lang == 'fortran':
            line = ('subroutine eval_cp (T, cp)\n\n'
                    # fortran needs type declarations
                    '  implicit none\n'
                    '  double precision, intent(in) :: T\n'
                    '  double precision, intent(out) :: cp({})\n'.format(num_s) +
                    '\n'
                    )
        elif lang == 'matlab':
            line = 'function cp = eval_cp (T)\n\n'
        file.write(line)

        # loop through species
        for isp, sp in enumerate(specs):
            line = '  if (T <= {:})'.format(sp.Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)

            line = '    ' + utils.get_array(lang, 'cp', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} + T * ('.format(sp.lo[0]) +
                     '{:.16e} + T * ('.format(sp.lo[1]) +
                     '{:.16e} + T * ('.format(sp.lo[2]) +
                     '{:.16e} + '.format(sp.lo[3]) +
                     '{:.16e} * T))))'.format(sp.lo[4]) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')

            line = '    ' + utils.get_array(lang, 'cp', isp)
            line += (' = {:.16e} * '.format(chem.RU / sp.mw) +
                     '({:.16e} + T * ('.format(sp.hi[0]) +
                     '{:.16e} + T * ('.format(sp.hi[1]) +
                     '{:.16e} + T * ('.format(sp.hi[2]) +
                     '{:.16e} + '.format(sp.hi[3]) +
                     '{:.16e} * T))))'.format(sp.hi[4]) +
                     utils.line_end[lang]
                     )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  }\n\n')
            elif lang == 'fortran':
                file.write('  end if\n\n')
            elif lang == 'matlab':
                file.write('  end\n\n')

        if lang in ['c', 'cuda']:
            file.write('} // end eval_cp\n\n')
        elif lang == 'fortran':
            file.write('end subroutine eval_cp\n\n')
        elif lang == 'matlab':
            file.write('end\n\n')

    return


def write_derivs(path, lang, specs, reacs):
    """Writes derivative function file and header.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in the mechanism.
    reacs : list of ReacInfo
        List of reactions in the mechanism.

    Returns
    -------
    None

    """

    pre = ''
    if lang == 'cuda': pre = '__device__ '

    # first write header file
    filename = 'dydt' + utils.header_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:
        file.write('#ifndef DYDT_HEAD\n'
                   '#define DYDT_HEAD\n'
                   '\n'
                   '#include "header{}"\n'.format(utils.header_ext[lang]) +
                   '\n'
                   '{0}void dydt (const double, const double, '
                   'const double*, double*);\n'
                   '\n'
                   '#endif\n'.format(pre)
                   )

    filename = 'dydt' + utils.file_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:
        file.write('#include "header{}"\n'.format(utils.header_ext[lang]))
        file.write('#include "chem_utils{0}"\n'
                   '#include "rates{0}"\n'.format(utils.header_ext[lang]))
        if lang == 'cuda':
            file.write('#include "gpu_memory.cuh"\n'
                       )
        file.write('\n')

        ##################################################################
        # constant pressure
        ##################################################################
        file.write('#if defined(CONP)\n\n')

        line = (pre + 'void dydt (const double t, const double pres, '
                      'const double * y, double * dy) {\n\n'
                )
        file.write(line)

        # calculation of species molar concentrations
        file.write('  // species molar concentrations\n'
                   '  double conc[{}];\n'.format(len(specs))
                   )
        file.write('  double y_N;\n')
        file.write('  double mw_avg;\n')
        file.write('  double rho;\n')

        # Simply call subroutine
        file.write('  eval_conc (' + utils.get_array(lang, 'y', 0) +
                   ', pres, &' + utils.get_array(lang, 'y', 1) + ', '
                   '&y_N, &mw_avg, &rho, conc);\n\n'
                   )

        # evaluate reaction rates
        rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
        file.write('  // local arrays holding reaction rates\n'
                   '  double fwd_rates[{}];\n'.format(len(reacs))
                   )
        if rev_reacs:
            file.write('  double rev_rates[{}];\n'.format(len(rev_reacs)))
        else:
            file.write('  double* rev_rates = 0;\n')
        file.write('  eval_rxn_rates (' + utils.get_array(lang, 'y', 0) +
                   ', pres, conc, fwd_rates, rev_rates);\n\n'
                   )

        # reaction pressure dependence
        num_dep_reacs = sum([rxn.thd_body or rxn.pdep for rxn in reacs])
        if num_dep_reacs > 0:
            file.write('  // get pressure modifications to reaction rates\n'
                       '  double pres_mod[{}];\n'.format(num_dep_reacs) +
                       '  get_rxn_pres_mod (' + utils.get_array(lang, 'y', 0) +
                       ', pres, conc, pres_mod);\n'
                       )
        else:
            file.write('  double* pres_mod = 0;\n')
        file.write('\n')

        # species rate of change of molar concentration
        file.write('  // evaluate species molar net production rates\n'
                   '  double dy_N;\n'
                   '  eval_spec_rates (fwd_rates, rev_rates, pres_mod, '
                   '&' + utils.get_array(lang, 'dy', 1) + ', &dy_N);\n\n'
                   )

        # evaluate specific heat
        file.write('  // local array holding constant pressure specific heat\n'
                   '  double cp[{}];\n'.format(len(specs)) +
                   '  eval_cp (' + utils.get_array(lang, 'y', 0) + ', cp);\n\n'
                   )

        file.write('  // constant pressure mass-average specific heat\n')
        line = '  double cp_avg = '
        isfirst = True
        for isp, sp in enumerate(specs[:-1]):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '             '

            if not isfirst: line += ' + '

            line += '(' + utils.get_array(lang, 'cp', isp) + \
                    ' * ' + utils.get_array(lang, 'y', isp + 1) + ')'

            isfirst = False

        if not isfirst: line += ' + '
        line += '(' + utils.get_array(lang, 'cp', len(specs) - 1) + ' * y_N)'
        file.write(line + utils.line_end[lang] + '\n')

        file.write('  // local array for species enthalpies\n'
                   '  double h[{}];\n'.format(len(specs)))
        file.write('  eval_h(' + utils.get_array(lang, 'y', 0) + ', h);\n')

        # energy equation
        file.write('  // rate of change of temperature\n')
        line = ('  ' + utils.get_array(lang, 'dy', 0) +
                ' = (-1.0 / (rho * cp_avg)) * ('
                )
        isfirst = True
        for isp, sp in enumerate(specs[:-1]):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '       '

            if not isfirst: line += ' + '

            line += ('(' + utils.get_array(lang, 'dy', isp + 1) + ' * ' +
                     utils.get_array(lang, 'h', isp) +
                     ' * {:.16e})'.format(sp.mw)
                     )

            isfirst = False
        line += (' + (dy_N * ' + utils.get_array(lang, 'h', len(specs) - 1) +
                 ' * {:.16e})'.format(specs[-1].mw)
                 )
        line += ')' + utils.line_end[lang] + '\n'
        file.write(line)

        # rate of change of species mass fractions
        file.write('  // calculate rate of change of species mass fractions\n')
        for idx, sp in enumerate(specs[:-1]):
            file.write('  ' + utils.get_array(lang, 'dy', idx + 1) +
                       ' *= ({:.16e} / rho);\n'.format(sp.mw)
                       )
        file.write('\n')

        file.write('} // end dydt\n\n')

        ##################################################################
        # constant volume
        ##################################################################
        file.write('#elif defined(CONV)\n\n')

        file.write(pre + 'void dydt (const double t, const double rho, '
                         'const double * y, double * dy) {\n\n'
                   )

        # calculation of species molar concentrations
        file.write('  // species molar concentrations\n'
                   '  double conc[{}];\n'.format(len(specs))
                   )

        file.write('  double y_N;\n')
        file.write('  double mw_avg;\n')
        file.write('  double pres;\n')

        # Simply call subroutine
        file.write('  eval_conc_rho (' + utils.get_array(lang, 'y', 0) +
                   'rho, &' + utils.get_array(lang, 'y', 1) + ', '
                   '&y_N, &mw_avg, &pres, conc);\n\n'
                   )

        # evaluate reaction rates
        rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
        file.write('  // local arrays holding reaction rates\n'
                   '  double fwd_rates[{}];\n'.format(len(reacs))
                   )
        if rev_reacs:
            file.write('  double rev_rates[{}];\n'.format(len(rev_reacs)))
        else:
            file.write('  double* rev_rates = 0;\n')
        file.write('  eval_rxn_rates (' + utils.get_array(lang, 'y', 0) + ', '
                   'pres, conc, fwd_rates, rev_rates);\n\n'
                   )

        # reaction pressure dependence
        num_dep_reacs = sum([rxn.thd_body or rxn.pdep for rxn in reacs])
        if num_dep_reacs > 0:
            file.write('  // get pressure modifications to reaction rates\n'
                       '  double pres_mod[{}];\n'.format(num_dep_reacs) +
                       '  get_rxn_pres_mod (' + utils.get_array(lang, 'y', 0) +
                       ', pres, conc, pres_mod);\n'
                       )
        else:
            file.write('  double* pres_mod = 0;\n')
        file.write('\n')

        # species rate of change of molar concentration
        file.write('  // evaluate species molar net production rates\n'
                   '  double dy_N;'
                   '  eval_spec_rates (fwd_rates, rev_rates, pres_mod, '
                   '&' + utils.get_array(lang, 'dy', 1) + ', &dy_N);\n\n'
                   )

        # evaluate specific heat
        file.write('  // local array holding constant volume specific heat\n'
                   '  double cv[{}];\n'.format(len(specs)) +
                   '  eval_cv(' +
                   utils.get_array(lang, 'y', 0) + ', cv);\n\n'
                   )

        file.write('  // constant volume mass-average specific heat\n')
        line = '  double cv_avg = '
        isfirst = True
        for idx, sp in enumerate(specs[:-1]):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '             '
            line += ' + ' if not isfirst else ''
            line += ('(' + utils.get_array(lang, 'cv', idx) + ' * ' +
                     utils.get_array(lang, 'y', idx + 1) + ')'
                     )

            isfirst = False
        line += '(' + utils.get_array(lang, 'cv', len(specs) - 1) + ' * y_N)'
        file.write(line + utils.line_end[lang] + '\n')

        # evaluate internal energy
        file.write('  // local array for species internal energies\n'
                   '  double u[{}];\n'.format(len(specs)) +
                   '  eval_u (' + utils.get_array(lang, 'y', 0) + ', u);\n\n'
                   )

        # energy equation
        file.write('  // rate of change of temperature\n')
        line = ('  ' + utils.get_array(lang, 'dy', 0) +
                ' = (-1.0 / (rho * cv_avg)) * ('
                )
        isfirst = True
        for idx, sp in enumerate(specs[:-1]):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '       '
            line += ' + ' if not isfirst else ''
            line += ('(' + utils.get_array(lang, 'dy', idx + 1) + ' * ' +
                     utils.get_array(lang, 'u', idx) +
                     ' * {:.16e})'.format(sp.mw)
                     )

            isfirst = False
        line += (' + (dy_N * ' + utils.get_array(lang, 'u', len(specs) - 1) +
                 ' * {:.16e})'.format(specs[-1].mw)
                 )
        line += ')' + utils.line_end[lang] + '\n'
        file.write(line)

        # rate of change of species mass fractions
        file.write('  // calculate rate of change of species mass fractions\n')
        for isp, sp in enumerate(specs[:-1]):
            file.write('  ' + utils.get_array(lang, 'dy', isp + 1) +
                       ' *= ({:.16e} / rho);\n'.format(sp.mw)
                       )

        file.write('\n')

        file.write('} // end dydt\n\n')
        file.write('#endif\n')

    return


def write_mass_mole(path, lang, specs):
    """Write files for mass/molar concentration and density conversion utility.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in mechanism.

    Returns
    -------
    None

    """

    # Create header file
    if lang in ['c', 'cuda']:
        filename = 'mass_mole{}'.format(utils.header_ext[lang])
        with open(os.path.join(path, filename), 'w') as file:
            file.write(
                '#ifndef MASS_MOLE_HEAD\n'
                '#define MASS_MOLE_HEAD\n'
                '\n'
                '#include "header{0}"\n'
                '\n'
                'void mole2mass (const double*, double*);\n'
                'void mass2mole (const double*, double*);\n'
                'double getDensity (const double, const double, const double*);\n'
                '\n'
                '#endif\n'.format(utils.header_ext[lang])
                )

    # Open file; both C and CUDA programs use C file (only used on host)
    filename = 'mass_mole' + utils.file_ext[lang]
    with open(os.path.join(path, filename), 'w') as file:
        if lang in ['c', 'cuda']:
            file.write('#include "mass_mole{}"\n\n'.format(
                utils.header_ext[lang]))

        ###################################################
        # Documentation and function/subroutine initializaton for mole2mass
        if lang in ['c', 'cuda']:
            file.write('/** Function converting species mole fractions to '
                       'mass fractions.\n'
                       ' *\n'
                       ' * \param[in]  X  array of species mole fractions\n'
                       ' * \param[out] Y  array of species mass fractions\n'
                       ' */\n'
                       'void mole2mass (const double * X, double * Y) {\n'
                       '\n'
                       )
        elif lang == 'fortran':
            file.write(
            '!-----------------------------------------------------------------\n'
            '!> Subroutine converting species mole fractions to mass fractions.\n'
            '!! @param[in]  X  array of species mole fractions\n'
            '!! @param[out] Y  array of species mass fractions\n'
            '!-----------------------------------------------------------------\n'
            'subroutine mole2mass (X, Y)\n'
            '  implicit none\n'
            '  double, dimension(:), intent(in) :: X\n'
            '  double, dimension(:), intent(out) :: X\n'
            '  double :: mw_avg\n'
            '\n'
            )

        file.write('  // mole fraction of final species\n')
        file.write(utils.line_start + 'double X_N' + utils.line_end[lang])
        line = '  X_N = 1.0 - ('
        isfirst = True
        for isp in range(len(specs) - 1):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '               '

            if not isfirst: line += ' + '

            line += utils.get_array(lang, 'X', isp)

            isfirst = False
        line += ')'
        file.write(line + utils.line_end[lang])

        # calculate molecular weight
        if lang in ['c', 'cuda']:
            file.write('  // average molecular weight\n'
                       '  double mw_avg = 0.0;\n'
                       )
            for isp in range(len(specs) - 1):
                sp = specs[isp]
                file.write('  mw_avg += X[{}] * '.format(isp) +
                           '{:.16e};\n'.format(sp.mw)
                           )
            file.write(utils.line_start + 'mw_avg += X_N * ' +
                           '{:.16e};\n'.format(specs[-1].mw)
                           )
        elif lang == 'fortran':
            file.write('  ! average molecular weight\n'
                       '  mw_avg = 0.0\n'
                       )
            for isp, sp in enumerate(specs):
                file.write('  mw_avg = mw_avg + '
                           'X({}) * '.format(isp + 1) +
                           '{:.16e}\n'.format(sp.mw)
                           )
        file.write('\n')

        # calculate mass fractions
        if lang in ['c', 'cuda']:
            file.write('  // calculate mass fractions\n')
            for isp in range(len(specs) - 1):
                sp = specs[isp]
                file.write('  Y[{0}] = X[{0}] * '.format(isp) +
                           '{:.16e} / mw_avg;\n'.format(sp.mw)
                           )
            file.write('\n'
                       '} // end mole2mass\n'
                       '\n'
                       )
        elif lang == 'fortran':
            file.write('  ! calculate mass fractions\n')
            for isp, sp in enumerate(specs):
                file.write('  Y({0}) = X({0}) * '.format(isp + 1) +
                           '{:.16e} / mw_avg\n'.format(sp.mw)
                           )
            file.write('\n'
                       'end subroutine mole2mass\n'
                       '\n'
                       )

        ################################
        # Documentation and function/subroutine initialization for mass2mole

        if lang in ['c', 'cuda']:
            file.write(
                '/** Function converting species mass fractions to mole '
                'fractions.\n'
                ' *\n'
                ' * \param[in]  Y  array of species mass fractions\n'
                ' * \param[out] X  array of species mole fractions\n'
                ' */\n'
                'void mass2mole (const double * Y, double * X) {\n'
                '\n'
                )
        elif lang == 'fortran':
            file.write(
                '!-------------------------------------------------------'
                '----------\n'
                '!> Subroutine converting species mass fractions to mole '
                'fractions.\n'
                '!! @param[in]  Y  array of species mass fractions\n'
                '!! @param[out] X  array of species mole fractions\n'
                '!-------------------------------------------------------'
                '----------\n'
                'subroutine mass2mole (Y, X)\n'
                '  implicit none\n'
                '  double, dimension(:), intent(in) :: Y\n'
                '  double, dimension(:), intent(out) :: X\n'
                '  double :: mw_avg\n'
                '\n'
                )

        # calculate Y_N
        file.write('  // mass fraction of final species\n')
        file.write(utils.line_start + 'double Y_N' + utils.line_end[lang])
        line = '  Y_N = 1.0 - ('
        isfirst = True
        for isp in range(len(specs) - 1):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '               '

            if not isfirst: line += ' + '

            line += utils.get_array(lang, 'Y', isp)

            isfirst = False
        line += ')'
        file.write(line + utils.line_end[lang])

        # calculate average molecular weight
        if lang in ['c', 'cuda']:
            file.write('  // average molecular weight\n')
            file.write('  double mw_avg = 0.0;\n')
            for isp in range(len(specs) - 1):
                file.write('  mw_avg += Y[{}] / '.format(isp) +
                           '{:.16e};\n'.format(specs[isp].mw)
                           )
            file.write('  mw_avg += Y_N / ' +
                           '{:.16e};\n'.format(specs[-1].mw)
                           )
            file.write('  mw_avg = 1.0 / mw_avg;\n')
        elif lang == 'fortran':
            file.write('  ! average molecular weight\n')
            file.write('  mw_avg = 0.0\n')
            for isp, sp in enumerate(specs):
                file.write('  mw_avg = mw_avg + '
                           'Y({}) / '.format(isp + 1) +
                           '{:.16e}\n'.format(sp.mw)
                           )
        file.write('\n')

        # calculate mole fractions
        if lang in ['c', 'cuda']:
            file.write('  // calculate mole fractions\n')
            for isp in range(len(specs) - 1):
                file.write('  X[{0}] = Y[{0}] * '.format(isp) +
                           'mw_avg / {:.16e};\n'.format(specs[isp].mw)
                           )
            file.write('\n'
                       '} // end mass2mole\n'
                       '\n'
                       )
        elif lang == 'fortran':
            file.write('  ! calculate mass fractions\n')
            for isp, sp in enumerate(specs):
                file.write('  X({0}) = Y({0}) * '.format(isp + 1) +
                           'mw_avg / {:.16e}\n'.format(sp.mw)
                           )
            file.write('\n'
                       'end subroutine mass2mole\n'
                       '\n'
                       )

        ###############################
        # Documentation and subroutine/function initialization for getDensity

        if lang in ['c', 'cuda']:
            file.write(
                '/** Function calculating density from mole fractions.\n'
                ' *\n'
                ' * \param[in]  temp  temperature\n'
                ' * \param[in]  pres  pressure\n'
                ' * \param[in]  X     array of species mole fractions\n'
                r' * \return     rho  mixture mass density' + '\n'
                ' */\n'
                'double getDensity (const double temp, const double '
                'pres, '
                'const double * X) {\n'
                '\n'
                )
        elif lang == 'fortran':
            file.write(
                '!-------------------------------------------------------'
                '----------\n'
                '!> Function calculating density from mole fractions.\n'
                '!! @param[in]  temp  temperature\n'
                '!! @param[in]  pres  pressure\n'
                '!! @param[in]  X     array of species mole fractions\n'
                '!! @return     rho   mixture mass density' + '\n'
                '!-------------------------------------------------------'
                '----------\n'
                'function mass2mole (temp, pres, X) result(rho)\n'
                '  implicit none\n'
                '  double, intent(in) :: temp, pres\n'
                '  double, dimension(:), intent(in) :: X\n'
                '  double :: mw_avg, rho\n'
                '\n'
                )

        file.write('  // mole fraction of final species\n')
        file.write(utils.line_start + 'double X_N' + utils.line_end[lang])
        line = '  X_N = 1.0 - ('
        isfirst = True
        for isp in range(len(specs) - 1):
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '               '

            if not isfirst: line += ' + '

            line += utils.get_array(lang, 'X', isp)

            isfirst = False
        line += ')'
        file.write(line + utils.line_end[lang])

        # get molecular weight
        if lang in ['c', 'cuda']:
            file.write('  // average molecular weight\n'
                       '  double mw_avg = 0.0;\n'
                       )
            for isp in range(len(specs) - 1):
                file.write('  mw_avg += X[{}] * '.format(isp) +
                           '{:.16e};\n'.format(specs[isp].mw)
                           )
            file.write(utils.line_start + 'mw_avg += X_N * ' +
                   '{:.16e};\n'.format(specs[-1].mw))
            file.write('\n')
        elif lang == 'fortran':
            file.write('  ! average molecular weight\n'
                       '  mw_avg = 0.0\n'
                       )
            for isp, sp in enumerate(specs):
                file.write('  mw_avg = mw_avg + '
                           'X({}) * '.format(isp + 1) +
                           '{:.16e}\n'.format(sp.mw)
                           )
            file.write('\n')

        # calculate density
        if lang in ['c', 'cuda']:
            file.write(
                '  return pres * mw_avg / ({:.8e} * temp);'.format(chem.RU)
                '\n'
                )
        else:
            line = '  rho = pres * mw_avg / ({:.8e} * temp)'.format(chem.RU)
            line += utils.line_end[lang]
            file.write(line)

        if lang in ['c', 'cuda']:
            file.write('} // end getDensity\n\n')
        elif lang == 'fortran':
            file.write('end function getDensity\n\n')

    return
