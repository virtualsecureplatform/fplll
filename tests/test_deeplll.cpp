/* Copyright (C) 2026

   This file is part of fplll. fplll is free software: you can redistribute it
   and/or modify it under the terms of the GNU Lesser General Public License. */

#include "test_utils.h"

using namespace std;
using namespace fplll;

template <class ZT>
bool is_deeplll_reduced(MatGSO<Z_NR<ZT>, FP_NR<mpfr_t>> &m, double delta, double eta, int depth)
{
  if (!m.update_gso())
    return false;

  FP_NR<mpfr_t> projected_norm, threshold, contribution, mu;
  for (int k = 0; k < m.d; k++)
  {
    for (int j = 0; j < k; j++)
    {
      m.get_mu(mu, k, j);
      mu.abs(mu);
      if (mu > eta)
        return false;
    }

    m.get_gram(projected_norm, k, k);
    for (int i = 0; i < k; i++)
    {
      if (i < depth || i >= k - depth)
      {
        threshold.mul(m.get_r_exp(i, i), delta);
        if (projected_norm < threshold)
          return false;
      }
      contribution.mul(m.get_mu_exp(k, i), m.get_mu_exp(k, i));
      contribution.mul(contribution, m.get_r_exp(i, i));
      projected_norm.sub(projected_norm, contribution);
    }
  }
  return true;
}

template <class ZT>
bool has_expected_transform(const ZZ_mat<ZT> &original, const ZZ_mat<ZT> &transform,
                            const ZZ_mat<ZT> &reduced)
{
  if (transform.get_rows() != original.get_rows() || reduced.get_rows() != original.get_rows() ||
      reduced.get_cols() != original.get_cols())
    return false;

  Z_NR<ZT> entry;
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
  int status         = 0;

  ZZ_mat<mpz_t> original(6, 7);
  original.gen_intrel(30);
  ZZ_mat<mpz_t> reduced = original;
  ZZ_mat<mpz_t> transform(1, 1);

  status = deeplll_reduction(reduced, transform, 4, LLL_DEF_DELTA, LLL_DEF_ETA, LM_PROVED, FT_MPFR,
                             192);
  if (status != RED_SUCCESS)
  {
    cerr << "DeepLLL reduction failed: " << get_red_status_str(status) << endl;
    FP_NR<mpfr_t>::set_prec(old_prec);
    return 1;
  }

  ZZ_mat<mpz_t> empty_u, empty_u_inv;
  MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> m(reduced, empty_u, empty_u_inv, GSO_INT_GRAM);
  if (!is_deeplll_reduced(m, LLL_DEF_DELTA, LLL_DEF_ETA, 4))
  {
    cerr << "Output of DeepLLL reduction is not DeepLLL reduced" << endl;
    status = 1;
  }
  if (!has_expected_transform(original, transform, reduced))
  {
    cerr << "DeepLLL transformation matrix does not reproduce the reduced basis" << endl;
    status = 1;
  }

  ZZ_mat<mpz_t> depth_zero = original;
  if (deeplll_reduction(depth_zero, 0, LLL_DEF_DELTA, LLL_DEF_ETA, LM_FAST, FT_DOUBLE) != RED_SUCCESS)
  {
    cerr << "Depth-zero DeepLLL reduction failed" << endl;
    status = 1;
  }
  MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> m_depth_zero(depth_zero, empty_u, empty_u_inv, GSO_INT_GRAM);
  if (!is_lll_reduced(m_depth_zero, LLL_DEF_DELTA, LLL_DEF_ETA))
  {
    cerr << "Depth-zero DeepLLL output is not LLL reduced" << endl;
    status = 1;
  }

  FP_NR<mpfr_t>::set_prec(old_prec);
  return status;
}
