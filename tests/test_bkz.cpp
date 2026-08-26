/* Copyright (C) 2016 Martin Albrecht

   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.

   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#include "test_utils.h"
#include <cstring>
#include <fplll/fplll.h>
#include <fplll/io/json.hpp>

using json = nlohmann::json;

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif

using namespace std;
using namespace fplll;

/**
   @brief Test BKZ reduction.

   @param A                test matrix
   @param block_size       block size
   @param float_type       floating point type to test
   @param flags            flags to use
   @param prec             precision if mpfr is used

   @return zero on success.
*/

template <class ZT>
int test_bkz(ZZ_mat<ZT> &A, const int block_size, FloatType float_type, int flags = BKZ_DEFAULT,
             int prec = 0)
{

  int status = 0;

  // zero on success
  status = bkz_reduction(A, block_size, flags, float_type, prec);
  if (status != RED_SUCCESS)
  {
    cerr << "BKZ reduction failed with error '" << get_red_status_str(status);
    cerr << " for float type " << FLOAT_TYPE_STR[float_type] << endl;
  }
  return status;
}

/**
   @brief Test BKZ strategy interface.

   @param A                test matrix
   @param block_size       block size

   @return zero on success.
*/

template <class ZT>
int test_bkz_param(ZZ_mat<ZT> &A, const int block_size, int flags = BKZ_DEFAULT,
                   string dump_gso_filename = string())
{

  int status = 0;

  vector<Strategy> strategies;
  for (long b = 0; b <= block_size; b++)
  {
    Strategy strategy = Strategy::EmptyStrategy(b);
    if (b == 10)
    {
      strategy.preprocessing_block_sizes.emplace_back(5);
    }
    else if (b == 20)
    {
      strategy.preprocessing_block_sizes.emplace_back(10);
    }
    else if (b == 30)
    {
      strategy.preprocessing_block_sizes.emplace_back(15);
    }
    strategies.emplace_back(std::move(strategy));
  }

  BKZParam params(block_size, strategies);
  params.flags             = flags;
  params.dump_gso_filename = dump_gso_filename;
  // zero on success
  status = bkz_reduction(&A, NULL, params, FT_DEFAULT, 53);
  if (status != RED_SUCCESS)
  {
    cerr << "BKZ parameter test failed with error '" << get_red_status_str(status) << "'" << endl;
  }
  return status;
}

/**
   @brief Test BKZ with pruning.

   @param A                test matrix
   @param block_size       block size

   @return zero on success.
*/

template <class ZT>
int test_bkz_param_linear_pruning(ZZ_mat<ZT> &A, const int block_size, int flags = BKZ_DEFAULT)
{

  int status = 0;
  vector<Strategy> strategies;
  for (long b = 0; b < block_size; b++)
  {
    Strategy strategy = Strategy::EmptyStrategy(b);
    if (b == 10)
    {
      strategy.preprocessing_block_sizes.emplace_back(5);
    }
    else if (b == 20)
    {
      strategy.preprocessing_block_sizes.emplace_back(10);
    }
    else if (b == 30)
    {
      strategy.preprocessing_block_sizes.emplace_back(15);
    }
    strategies.emplace_back(std::move(strategy));
  }

  Strategy strategy;
  strategy.pruning_parameters.emplace_back(
      PruningParams::LinearPruningParams(block_size, block_size / 2));
  strategies.emplace_back(std::move(strategy));

  BKZParam params(block_size, strategies);
  params.flags = flags;
  // zero on success
  status = bkz_reduction(&A, NULL, params, FT_DEFAULT, 53);
  if (status != RED_SUCCESS)
  {
    cerr << "BKZ parameter test failed with error '" << get_red_status_str(status) << "'" << endl;
  }
  return status;
}

template <class ZT>
int test_bkz_param_pruning(ZZ_mat<ZT> &A, const int block_size, int flags = BKZ_DEFAULT)
{

  int status = 0;

  vector<Strategy> strategies = load_strategies_json(TESTDATADIR "/strategies/default.json");
  BKZParam params(block_size, strategies);
  params.flags = flags;
  // zero on success
  status = bkz_reduction(&A, NULL, params, FT_DEFAULT, 53);
  if (status != RED_SUCCESS)
  {
    cerr << "BKZ parameter test failed with error '" << get_red_status_str(status) << "'" << endl;
  }
  return status;
}

/**
   @brief Test BKZ_DUMP_GSO for a matrix d × (d+1) integer relations matrix with bit size b (copied
   from test_int_rel)

   @param d                dimension
   @param b                bit size
   @param block_size       block size
   @param float_type       floating point type to test
   @param flags            flags to use

   @return zero on success.
 */
template <class ZT>
int test_int_rel_bkz_dump_gso(int d, int b, const int block_size,
                              int flags = BKZ_DEFAULT | BKZ_DUMP_GSO)
{
  ZZ_mat<ZT> A, B;
  A.resize(d, d + 1);
  A.gen_intrel(b);
  B          = A;
  int status = 0;
  // TODO: maybe not safe.
  string file_bkz_dump_gso = tmpnam(nullptr);
  status |= test_bkz_param<ZT>(B, block_size, flags, file_bkz_dump_gso);

  if (status != 0)
  {
    cerr << "Error in test_bkz_param." << endl;
    return status;
  }

  json js;
  std::ifstream fs(file_bkz_dump_gso);
  if (fs.fail())
  {
    cerr << "File " << file_bkz_dump_gso << " cannot be loaded." << endl;
    return 1;
  }
  fs >> js;
  int loop    = -1;
  double time = 0.0;
  for (auto i : js)
  {
    // Verify if there are as much norms as there are rows in A
    if (A.get_rows() != (int)i["norms"].size())
    {
      cerr << "The array \"norms\" does not contain enough values (" << A.get_rows()
           << " expected but " << i["norms"].size() << " found)." << endl;
      return 1;
    }

    // Extract data from json file
    const string step_js = i["step"];
    const int loop_js    = i["loop"];
    const double time_js = i["time"];
    // Verify if loop of Input and Output have loop = -1
    if (step_js.compare("Input") == 0 || step_js.compare("Output") == 0)
    {
      if (loop_js != -1)
      {
        cerr << "Steps Input or Output are not with \"loop\" = -1." << endl;
        return 1;
      }
    }
    else
    {
      // Verify that loop increases
      loop++;
      if (loop_js != loop)
      {
        cerr << "Loop does not increase." << endl;
        return 1;
      }
      // Verify that time increases
      if (time > time_js)
      {
        cerr << "Time does not increase." << endl;
        return 1;
      }
      time = time_js;
    }
  }

  return 0;
}

/**
   @brief Test BKZ for matrix stored in file pointed to by `input_filename`.

   @param input_filename   a path
   @param block_size       block size
   @param float_type       floating point type to test
   @param flags            flags to use
   @param prec             precision if mpfr is used

   @return zero on success
*/

template <class ZT>
int test_filename(const char *input_filename, const int block_size,
                  FloatType float_type = FT_DEFAULT, int flags = BKZ_DEFAULT, int prec = 0)
{
  ZZ_mat<ZT> A, B;
  int status = 0;
  status |= read_file(A, input_filename);
  B = A;
  status |= test_bkz<ZT>(A, block_size, float_type, flags, prec);
  status |= test_bkz_param<ZT>(B, block_size);
  return status;
}

/**
   @brief Construct d × (d+1) integer relations matrix with bit size b and test BKZ.

   @param d                dimension
   @param b                bit size
   @param block_size       block size
   @param float_type       floating point type to test
   @param flags            flags to use
   @param prec             precision if mpfr is used

   @return zero on success
*/

template <class ZT>
int test_int_rel(int d, int b, const int block_size, FloatType float_type = FT_DEFAULT,
                 int flags = BKZ_DEFAULT, int prec = 0)
{
  ZZ_mat<ZT> A, B;
  A.resize(d, d + 1);
  A.gen_intrel(b);
  B          = A;
  int status = 0;
  status |= test_bkz<ZT>(A, block_size, float_type, flags | BKZ_VERBOSE, prec);
  status |= test_bkz_param<ZT>(B, block_size);
  status |= test_bkz_param_linear_pruning<ZT>(B, block_size);
  status |= test_bkz_param_pruning<ZT>(B, block_size);
  return status;
}

int test_linear_dep()
{
  ZZ_mat<mpz_t> A;
  std::stringstream("[[1 2 3]\n [4 5 6]\n [7 8 9]]\n") >> A;
  return test_bkz_param<mpz_t>(A, 3);
}

int test_bkz_jump()
{
  ZZ_mat<mpz_t> A;
  if (read_file(A, TESTDATADIR "/tests/lattices/example_in") != 0)
    return 1;

  vector<Strategy> strategies;
  BKZParam param(5, strategies);
  param.tour_step = 2;
  param.flags |= BKZ_MAX_LOOPS;
  param.max_loops = 2;
  const int status = bkz_reduction(&A, NULL, param, FT_DOUBLE, 0);
  return (status == RED_SUCCESS || status == RED_BKZ_LOOPS_LIMIT) ? 0 : 1;
}

int test_bkz_truncated_tours()
{
  ZZ_mat<mpz_t> A;
  if (read_file(A, TESTDATADIR "/tests/lattices/dim55_in") != 0)
    return 1;

  vector<Strategy> strategies;
  BKZParam param(10, strategies);
  param.max_tour_rows = 30;
  param.flags |= BKZ_MAX_LOOPS;
  param.max_loops = 2;
  const int status = bkz_reduction(&A, NULL, param, FT_DOUBLE, 0);
  return (status == RED_SUCCESS || status == RED_BKZ_LOOPS_LIMIT) ? 0 : 1;
}

int test_progressive_bkz()
{
  ZZ_mat<mpz_t> A;
  if (read_file(A, TESTDATADIR "/tests/lattices/example_in") != 0)
    return 1;

  vector<Strategy> strategies;
  for (int block = 0; block <= 5; ++block)
    strategies.emplace_back(Strategy::EmptyStrategy(block));
  BKZParam small(3, strategies);
  BKZParam large(5, strategies);
  small.flags = large.flags = BKZ_MAX_LOOPS;
  small.max_loops = large.max_loops = 1;
  vector<BKZParam> stages{small, large};
  const int status = progressive_bkz_reduction(&A, stages, FT_DOUBLE, 0);
  return status == RED_SUCCESS ? 0 : 1;
}

int test_scoped_row_transform()
{
  ZZ_mat<mpz_t> basis(4, 4), original, u, u_inv_t;
  basis.gen_identity(4);
  basis(1, 0) = 3;
  basis(2, 1) = 5;
  basis(3, 2) = 7;
  original = basis;
  u.gen_identity(4);

  MatGSO<Z_NR<mpz_t>, FP_NR<double>> gso(basis, u, u_inv_t, GSO_DEFAULT);
  gso.discover_all_rows();
  RowTransform<Z_NR<mpz_t>> transform;
  RowTransform<Z_NR<mpz_t>> nested_transform;
  {
    ScopedRowTransformRecorder<Z_NR<mpz_t>, FP_NR<double>> recorder(gso, 1, 4, transform);
    FP_NR<double> coefficient;
    coefficient = 6;
    gso.row_op_begin(1, 4);
    gso.row_addmul(3, 1, coefficient);
    gso.row_op_end(1, 4);
    {
      ScopedRowTransformRecorder<Z_NR<mpz_t>, FP_NR<double>> nested(gso, 1, 3,
                                                                    nested_transform);
      gso.negate_row_of_b(2);
    }
    gso.move_row(3, 1);
  }

  if (transform.empty() || transform.dimension() != 3)
    return 1;
  if (nested_transform.dimension() != 2 || nested_transform.matrix()(0, 0) != 1 ||
      nested_transform.matrix()(1, 1) != -1)
    return 1;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < basis.get_cols(); ++column)
    {
      Z_NR<mpz_t> expected;
      expected = 0;
      for (int source = 0; source < 3; ++source)
        expected.addmul(transform.matrix()(row, source), original(1 + source, column));
      if (basis(1 + row, column) != expected)
        return 1;
      const Z_NR<mpz_t> expected_u = column == 0 ? Z_NR<mpz_t>()
                                                  : transform.matrix()(row, column - 1);
      if (u(1 + row, column) != expected_u)
        return 1;
    }

  return gso.update_gso() ? 0 : 1;
}

int test_bounded_preprocessing_transform()
{
  ZZ_mat<mpz_t> basis(4, 4), original, u, u_inv_t;
  basis.gen_zero(4, 4);
  basis(0, 0) = 1;
  basis(1, 0) = 5;
  basis(1, 1) = 4;
  basis(2, 0) = 2;
  basis(2, 1) = 1;
  basis(2, 2) = 1;
  basis(3, 0) = 3;
  basis(3, 3) = 1;
  original = basis;
  u.gen_identity(4);

  MatGSO<Z_NR<mpz_t>, FP_NR<double>> gso(basis, u, u_inv_t, GSO_DEFAULT);
  gso.discover_all_rows();
  if (!gso.update_gso())
    return 1;
  LLLReduction<Z_NR<mpz_t>, FP_NR<double>> lll(gso, LLL_DEF_DELTA, LLL_DEF_ETA,
                                               LLL_DEFAULT);
  vector<Strategy> strategies;
  BKZParam param(3, strategies);
  param.flags = BKZ_BOUNDED_LLL;
  BKZReduction<Z_NR<mpz_t>, FP_NR<double>> bkz(gso, lll, param);
  RowTransform<Z_NR<mpz_t>> transform;
  bkz.svp_preprocessing(1, 3, param, &transform);

  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 4; ++column)
    {
      Z_NR<mpz_t> expected;
      expected = 0;
      for (int source = 0; source < 3; ++source)
        expected.addmul(transform.matrix()(row, source), original(1 + source, column));
      if (basis(1 + row, column) != expected)
        return 1;
      const Z_NR<mpz_t> expected_u = column == 0 ? Z_NR<mpz_t>()
                                                  : transform.matrix()(row, column - 1);
      if (u(1 + row, column) != expected_u)
        return 1;
    }
  for (int column = 0; column < 4; ++column)
    if (basis(0, column) != original(0, column))
      return 1;
  return gso.update_gso() ? 0 : 1;
}

int test_local_block_transform()
{
  ZZ_mat<mpz_t> A(4, 4), U, UT;
  A.gen_identity(4);
  A(1, 0) = 2;
  A(2, 1) = 3;
  A(3, 2) = 5;
  ZZ_mat<mpz_t> expected = A;

  LocalBlockTransform transform(2);
  Z_NR<mpz_t> large_coefficient;
  large_coefficient = 1;
  large_coefficient.mul_2si(large_coefficient, 80);
  transform.row_addmul(1, 0, large_coefficient);
  transform.negate_row(0);
  transform.move_row(0, 1);

  expected[2].addmul(expected[1], large_coefficient);
  for (int j = 0; j < expected.get_cols(); ++j)
    expected(1, j).neg(expected(1, j));
  expected.rotate_left(1, 2);

  MatGSO<Z_NR<mpz_t>, FP_NR<double>> gso(A, U, UT, GSO_DEFAULT);
  gso.discover_all_rows();
  gso.apply_integer_transform(transform.matrix(), 1);
  if (transform.matrix().get_rows() != 2)
    return 1;
  for (int i = 0; i < A.get_rows(); ++i)
    for (int j = 0; j < A.get_cols(); ++j)
      if (A(i, j) != expected(i, j))
        return 1;
  return gso.update_gso() ? 0 : 1;
}

int test_local_postprocessing()
{
  ZZ_mat<mpz_t> A(4, 4), U, UT;
  A.gen_identity(4);
  A(2, 0) = 7;
  U.gen_identity(4);
  MatGSO<Z_NR<mpz_t>, FP_NR<double>> gso(A, U, UT, GSO_DEFAULT);
  gso.discover_all_rows();
  LLLReduction<Z_NR<mpz_t>, FP_NR<double>> lll(gso, LLL_DEF_DELTA, LLL_DEF_ETA, LLL_DEFAULT);
  vector<Strategy> strategies;
  BKZParam param(2, strategies);
  BKZReduction<Z_NR<mpz_t>, FP_NR<double>> bkz(gso, lll, param);

  LocalBlockTransform transform(2);
  transform.row_addmul_si(1, 0, 3);
  if (bkz.local_postprocessing(1, 2, transform))
    return 1;
  return gso.update_gso() ? 0 : 1;
}

int main(int /*argc*/, char ** /*argv*/)
{

  int status = 0;

  status |= test_linear_dep();
  status |= test_bkz_jump();
  status |= test_bkz_truncated_tours();
  status |= test_progressive_bkz();
  status |= test_scoped_row_transform();
  status |= test_bounded_preprocessing_transform();
  status |= test_local_block_transform();
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", 10, FT_DEFAULT,
                                 BKZ_DEFAULT | BKZ_AUTO_ABORT);
#ifdef FPLLL_WITH_QD
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", 10, FT_DD,
                                 BKZ_SD_VARIANT | BKZ_AUTO_ABORT);
#endif
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", 10, FT_DEFAULT, BKZ_SLD_RED);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", 20, FT_MPFR,
                                 BKZ_DEFAULT | BKZ_AUTO_ABORT, 128);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", 20, FT_MPFR,
                                 BKZ_SD_VARIANT | BKZ_AUTO_ABORT, 128);
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/dim55_in", 20, FT_MPFR, BKZ_SLD_RED, 128);

  status |= test_int_rel<mpz_t>(50, 1000, 10, FT_DOUBLE, BKZ_DEFAULT | BKZ_AUTO_ABORT);
  status |= test_int_rel<mpz_t>(50, 1000, 10, FT_DOUBLE, BKZ_SD_VARIANT | BKZ_AUTO_ABORT);
  status |= test_int_rel<mpz_t>(50, 1000, 10, FT_DOUBLE, BKZ_SLD_RED);
  status |= test_int_rel<mpz_t>(50, 1000, 15, FT_MPFR, BKZ_DEFAULT | BKZ_AUTO_ABORT, 100);
  status |= test_int_rel<mpz_t>(50, 1000, 15, FT_MPFR, BKZ_SD_VARIANT | BKZ_AUTO_ABORT, 100);
  status |= test_int_rel<mpz_t>(50, 1000, 15, FT_MPFR, BKZ_SLD_RED, 100);

  status |= test_int_rel<mpz_t>(30, 2000, 10, FT_DPE, BKZ_DEFAULT | BKZ_AUTO_ABORT);
  status |= test_int_rel<mpz_t>(30, 2000, 10, FT_DPE, BKZ_SD_VARIANT | BKZ_AUTO_ABORT);
  status |= test_int_rel<mpz_t>(30, 2000, 10, FT_DPE, BKZ_SLD_RED);
  status |= test_int_rel<mpz_t>(30, 2000, 10, FT_MPFR, BKZ_DEFAULT | BKZ_AUTO_ABORT, 53);
  status |= test_int_rel<mpz_t>(30, 2000, 10, FT_MPFR, BKZ_SD_VARIANT | BKZ_AUTO_ABORT, 53);
  status |= test_int_rel<mpz_t>(30, 2000, 10, FT_MPFR, BKZ_SLD_RED, 53);

  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DEFAULT,
                                 BKZ_SD_VARIANT);
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DEFAULT, BKZ_SLD_RED);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DOUBLE);
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DOUBLE, BKZ_SD_VARIANT);
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DOUBLE, BKZ_SLD_RED);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_MPFR,
                                 BKZ_AUTO_ABORT, 212);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_MPFR,
                                 BKZ_SD_VARIANT | BKZ_AUTO_ABORT, 212);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_MPFR,
                                 BKZ_SLD_RED | BKZ_AUTO_ABORT, 212);
  status |= test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DOUBLE);
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DOUBLE, BKZ_SD_VARIANT);
  status |=
      test_filename<mpz_t>(TESTDATADIR "/tests/lattices/example_in", 10, FT_DOUBLE, BKZ_SLD_RED);

  // Test BKZ_DUMP_GSO
  status |= test_int_rel_bkz_dump_gso<mpz_t>(50, 1000, 15, BKZ_DEFAULT | BKZ_DUMP_GSO);

  if (status == 0)
  {
    cerr << "All tests passed." << endl;
    return 0;
  }
  else
  {
    return -1;
  }

  return 0;
}
