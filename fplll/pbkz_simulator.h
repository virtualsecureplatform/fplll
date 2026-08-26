/* Progressive BKZ quality simulator (Aono--Nguyen--Shen, 2016). */
#ifndef FPLLL_PBKZ_SIMULATOR_H
#define FPLLL_PBKZ_SIMULATOR_H

#include "defs.h"
#include <vector>

FPLLL_BEGIN_NAMESPACE

/** Natural logarithm of the full enumeration cost (FEC), up to the
 * conventional factor for sign symmetry.  Entries are log Gram--Schmidt
 * lengths, not squared lengths. */
long double pbkz_log_fec(const std::vector<long double> &log_gso_lengths);

/** Simulate the log Gram--Schmidt lengths of a converged BKZ-beta basis.
 * The returned determinant is normalized to one. */
std::vector<long double> pbkz_simulate_gso(int dimension, int block_size);

/** Paper threshold Sim-FEC(n, beta). */
long double pbkz_simulated_log_fec(int dimension, int block_size);

FPLLL_END_NAMESPACE

#endif
