/* Copyright (C) 2026

   This file is part of fplll. fplll is free software: you can redistribute it
   and/or modify it under the terms of the GNU Lesser General Public License. */

#include "test_utils.h"

using namespace std;
using namespace fplll;

bool is_potlll_reduced(MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> &m, double delta, double eta)
{
  if (!m.update_gso())
    return false;

  FP_NR<mpfr_t> potential, projected_norm, contribution, quotient, mu;
  for (int k = 0; k < m.d; k++)
  {
    for (int j = 0; j < k; j++)
    {
      m.get_mu(mu, k, j);
      mu.abs(mu);
      if (mu > eta)
        return false;
    }

    potential     = 1;
    projected_norm = m.get_r_exp(k, k);
    for (int j = k - 1; j >= 0; j--)
    {
      contribution.mul(m.get_mu_exp(k, j), m.get_mu_exp(k, j));
      contribution.mul(contribution, m.get_r_exp(j, j));
      projected_norm.add(projected_norm, contribution);
      quotient.div(projected_norm, m.get_r_exp(j, j));
      potential.mul(potential, quotient);
      if (potential < delta)
        return false;
    }
  }
  return true;
}

bool has_expected_transform(const ZZ_mat<mpz_t> &original, const ZZ_mat<mpz_t> &transform,
                            const ZZ_mat<mpz_t> &reduced)
{
  Z_NR<mpz_t> entry;
  for (int i = 0; i < reduced.get_rows(); i++)
  {
    for (int j = 0; j < reduced.get_cols(); j++)
    {
      entry = 0;
      for (int k = 0; k < original.get_rows(); k++)
        entry.addmul(transform(i, k), original(k, j));
      if (entry != reduced(i, j))
        return false;
    }
  }
  return true;
}

int main()
{
  const int old_prec = FP_NR<mpfr_t>::set_prec(256);
  ZZ_mat<mpz_t> original(6, 7);
  original.gen_intrel(30);
  ZZ_mat<mpz_t> reduced = original;
  ZZ_mat<mpz_t> transform(1, 1);

  const int status = potlll_reduction(reduced, transform, LLL_DEF_DELTA, LLL_DEF_ETA, LM_PROVED,
                                      FT_MPFR, 192);
  if (status != RED_SUCCESS)
  {
    cerr << "PotLLL reduction failed: " << get_red_status_str(status) << endl;
    FP_NR<mpfr_t>::set_prec(old_prec);
    return 1;
  }

  ZZ_mat<mpz_t> empty_u, empty_u_inv;
  MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> m(reduced, empty_u, empty_u_inv, GSO_INT_GRAM);
  if (!is_potlll_reduced(m, LLL_DEF_DELTA, LLL_DEF_ETA))
  {
    cerr << "Output of PotLLL reduction is not PotLLL reduced" << endl;
    FP_NR<mpfr_t>::set_prec(old_prec);
    return 1;
  }
  if (!has_expected_transform(original, transform, reduced))
  {
    cerr << "PotLLL transformation matrix does not reproduce the reduced basis" << endl;
    FP_NR<mpfr_t>::set_prec(old_prec);
    return 1;
  }

  /* Exercise the row-exponent scaling used by the fast backend. */
  ZZ_mat<mpz_t> fast_reduced = original;
  ZZ_mat<mpz_t> fast_transform(1, 1);
  const int fast_status =
      potlll_reduction(fast_reduced, fast_transform, LLL_DEF_DELTA, LLL_DEF_ETA, LM_FAST);
  MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> fast_m(fast_reduced, empty_u, empty_u_inv, GSO_INT_GRAM);
  if (fast_status != RED_SUCCESS || !is_potlll_reduced(fast_m, LLL_DEF_DELTA, LLL_DEF_ETA) ||
      !has_expected_transform(original, fast_transform, fast_reduced))
  {
    cerr << "Fast PotLLL reduction failed" << endl;
    FP_NR<mpfr_t>::set_prec(old_prec);
    return 1;
  }

  FP_NR<mpfr_t>::set_prec(old_prec);
  return 0;
}
