/*
 * Progressive BKZ simulator from Aono, Nguyen and Shen, "Lattice
 * Enumeration Using Extreme Pruning", ePrint 2016/146, Sec. 4.  The
 * parameter fit and small-block correction follow the authors' LGPL
 * reference implementation pbkzlib-202205.
 */
#include "pbkz_simulator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

FPLLL_BEGIN_NAMESPACE

namespace
{
long double log_ball_volume(int dimension)
{
  return 0.5L * dimension * std::log(acosl(-1.0L)) - lgammal(0.5L * dimension + 1.0L);
}

long double log_add(long double lhs, long double rhs)
{
  if (lhs < rhs)
    std::swap(lhs, rhs);
  if (!std::isfinite(rhs))
    return lhs;
  return lhs + log1pl(expl(rhs - lhs));
}

long double beta_fraction(long double a, long double b, long double x)
{
  const long double tiny = std::numeric_limits<long double>::min() * 1e4L;
  long double qab = a + b, qap = a + 1.0L, qam = a - 1.0L;
  long double c = 1.0L, d = 1.0L - qab * x / qap;
  if (fabsl(d) < tiny)
    d = tiny;
  d = 1.0L / d;
  long double h = d;
  for (int m = 1; m <= 200; ++m)
  {
    const int m2 = 2 * m;
    long double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0L + aa * d;
    c = 1.0L + aa / c;
    if (fabsl(d) < tiny) d = tiny;
    if (fabsl(c) < tiny) c = tiny;
    d = 1.0L / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0L + aa * d;
    c = 1.0L + aa / c;
    if (fabsl(d) < tiny) d = tiny;
    if (fabsl(c) < tiny) c = tiny;
    d = 1.0L / d;
    const long double delta = d * c;
    h *= delta;
    if (fabsl(delta - 1.0L) < 1e-16L)
      break;
  }
  return h;
}

long double regularized_beta(long double a, long double b, long double x)
{
  if (x <= 0.0L) return 0.0L;
  if (x >= 1.0L) return 1.0L;
  const long double front = expl(lgammal(a + b) - lgammal(a) - lgammal(b) +
                                  a * logl(x) + b * log1pl(-x));
  if (x < (a + 1.0L) / (a + b + 2.0L))
    return front * beta_fraction(a, b, x) / a;
  return 1.0L - front * beta_fraction(b, a, 1.0L - x) / b;
}

long double inverse_regularized_beta(long double a, long double b, long double probability)
{
  long double low = 0.0L, high = 1.0L;
  for (int iteration = 0; iteration < 100; ++iteration)
  {
    const long double middle = 0.5L * (low + high);
    if (regularized_beta(a, b, middle) < probability)
      low = middle;
    else
      high = middle;
  }
  return 0.5L * (low + high);
}

long double small_block_alpha(int dimension)
{
  static const long double cn[] = {
      0.593208L,  0.582161L,  0.561454L,  0.544344L,  0.522066L,  0.502545L,
      0.479628L,  0.459819L,  0.438675L,  0.413708L,  0.392483L,  0.370717L,
      0.344447L,  0.322574L,  0.297318L,  0.273761L,  0.249247L,  0.225483L,
      0.199940L,  0.173832L,  0.147417L,  0.123425L,  0.100035L,  0.074487L,
      0.043089L,  0.020321L, -0.013844L, -0.042863L, -0.068204L, -0.093892L,
     -0.124345L, -0.151097L, -0.183912L, -0.214122L, -0.241654L, -0.274612L,
     -0.302966L, -0.330965L, -0.367514L, -0.391956L, -0.426507L, -0.457813L,
     -0.488113L, -0.518525L, -0.554184L, -0.585479L, -0.617705L, -0.646749L,
     -0.671864L, -0.687300L};
  if (dimension < 1 || dimension > 49)
    return 0.0L;
  long double sum = 0.0L;
  for (int j = 0; j < dimension; ++j)
    sum += cn[49 - j];
  const long double log_det = sum / dimension - log_ball_volume(dimension) / dimension;
  return expl(cn[50 - dimension] - log_det);
}

long double target_log_r(int beta)
{
  const long double fit1 = -18.2139L / (beta + 318.978L);
  long double result     = fit1;
  if (beta > 90)
    result = std::max(fit1, -1.06889L / (beta - 31.0345L) *
                                logl(0.417419L * beta - 25.4889L));
  if (beta <= 35)
    result -= 2.0L * logl(small_block_alpha(beta)) / beta;
  return result;
}

long double expected_alpha(int beta)
{
  const long double log_r = target_log_r(beta);
  return expl(logl((beta + 1.0L) / beta) + log_ball_volume(beta) / beta -
              (beta - 1.0L) * log_r / 4.0L);
}

void set_gh_value(std::vector<long double> &logs, int begin, int end, long double alpha)
{
  const int dimension = end - begin + 1;
  long double tail    = 0.0L;
  for (int i = begin + 1; i <= end; ++i)
    tail += logs[i];
  logs[begin] = tail / (dimension - 1.0L) - log_ball_volume(dimension) /
                                                       (dimension - 1.0L) +
                dimension * logl(alpha) / (dimension - 1.0L);
}

long double log_enum_cost(const std::vector<long double> &logs, int begin, int end,
                          long double alpha, long double probability)
{
  const int dimension = end - begin + 1;
  long double log_det = 0.0L;
  for (int i = begin; i <= end; ++i)
    log_det += logs[i];
  const long double log_radius = logl(alpha) + log_det / dimension -
                                 log_ball_volume(dimension) / dimension;
  long double accumulated = 0.0L;
  long double result = -std::numeric_limits<long double>::infinity();
  for (int i = end; i >= begin + 2; --i)
  {
    accumulated += log_radius - logs[i];
    const long double depth = end - i + 1.0L;
    const long double x = inverse_regularized_beta(0.5L * depth, 0.5L * (i - begin),
                                                   probability);
    result = log_add(result, accumulated + log_ball_volume(static_cast<int>(depth)) +
                                 0.5L * depth * logl(x));
  }
  return result - logl(2.0L);
}
} // namespace

long double pbkz_log_fec(const std::vector<long double> &logs)
{
  const int dimension = static_cast<int>(logs.size());
  if (dimension == 0)
    return -std::numeric_limits<long double>::infinity();
  long double log_det = 0.0L;
  for (long double value : logs)
    log_det += value;
  const long double log_gh = log_det / dimension - log_ball_volume(dimension) / dimension;
  long double suffix = 0.0L;
  long double result = -std::numeric_limits<long double>::infinity();
  for (int k = 1; k <= dimension; ++k)
  {
    suffix += logs[dimension - k];
    result = log_add(result, log_ball_volume(k) + k * log_gh - suffix);
  }
  return result - logl(2.0L);
}

std::vector<long double> pbkz_simulate_gso(int dimension, int beta)
{
  if (dimension < 2 || beta < 2 || beta > dimension)
    throw std::invalid_argument("invalid progressive BKZ simulator dimensions");
  const long double alpha = expected_alpha(beta);
  const long double probability = std::min(1.0L, 2.0L / powl(alpha, beta));
  const long double points = 0.5L * probability * powl(alpha, beta);
  const long double full_log_ratio = logl(points) + lgammal(points) +
                                     lgammal((1.0L + beta) / beta) -
                                     lgammal(points + (1.0L + beta) / beta);
  long double full_found_alpha = alpha * expl(full_log_ratio);
  if (beta < 50)
    full_found_alpha = std::max(full_found_alpha, small_block_alpha(beta));
  std::vector<long double> logs(dimension, 0.0L);
  for (int i = dimension - 2; i >= 0; --i)
  {
    const int local_beta = std::min(beta, dimension - i);
    const long double points = 0.5L * probability * powl(alpha, local_beta);
    const long double log_ratio = logl(points) + lgammal(points) +
                                  lgammal((1.0L + local_beta) / local_beta) -
                                  lgammal(points + (1.0L + local_beta) / local_beta);
    long double found_alpha = alpha * expl(log_ratio);
    if (local_beta < 50)
      found_alpha = std::max(found_alpha, small_block_alpha(local_beta));
    set_gh_value(logs, i, i + local_beta - 1, found_alpha);
  }

  // Sharp-simulator second phase: the last incomplete blocks use a larger
  // success probability chosen to match the predicted cost of a full block.
  vector<long double> found_alphas(dimension, full_found_alpha);
  const int last_full = dimension - beta;
  const long double base_cost = log_enum_cost(logs, last_full, dimension - 1, alpha,
                                               probability);
  for (int i = 0; i <= last_full; ++i)
    found_alphas[i] = full_found_alpha;
  for (int i = last_full + 1; i < dimension - 1; ++i)
  {
    const int local_beta = dimension - i;
    long double low = probability, high = 0.5L;
    if (low > high)
      low = high;
    for (int iteration = 0; iteration < 80; ++iteration)
    {
      const long double candidate = 0.5L * (low + high);
      const long double used_alpha = powl(0.5L * candidate, -1.0L / local_beta);
      if (log_enum_cost(logs, i, dimension - 1, used_alpha, candidate) < base_cost)
        low = candidate;
      else
        high = candidate;
    }
    const long double tail_probability = 0.5L * (low + high);
    const long double used_alpha = powl(0.5L * tail_probability, -1.0L / local_beta);
    long double tail_found = used_alpha * local_beta / (local_beta + 1.0L);
    if (local_beta < 50)
      tail_found = std::max(tail_found, small_block_alpha(local_beta));
    else
      tail_found = std::max(tail_found, expl(2.77258872223978L / local_beta));
    found_alphas[i] = tail_found;
  }
  logs.assign(dimension, 0.0L);
  for (int i = dimension - 2; i >= 0; --i)
    set_gh_value(logs, i, std::min(dimension - 1, i + beta - 1), found_alphas[i]);
  long double mean = 0.0L;
  for (long double value : logs)
    mean += value / dimension;
  for (long double &value : logs)
    value -= mean;
  return logs;
}

long double pbkz_simulated_log_fec(int dimension, int beta)
{
  return pbkz_log_fec(pbkz_simulate_gso(dimension, beta));
}

FPLLL_END_NAMESPACE
