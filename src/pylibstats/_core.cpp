/// pylibstats native extension module.
///
/// Binds all libstats distribution classes to Python via nanobind, exposing
/// scalar and SIMD-accelerated batch operations through NumPy arrays.

#include <nanobind/nanobind.h>

#include <limits>

// libstats distribution headers
#include <libstats/distributions/bernoulli.h>
#include <libstats/distributions/beta.h>
#include <libstats/distributions/binomial.h>
#include <libstats/distributions/cauchy.h>
#include <libstats/distributions/chi_squared.h>
#include <libstats/distributions/discrete.h>
#include <libstats/distributions/erlang.h>
#include <libstats/distributions/exponential.h>
#include <libstats/distributions/fisher_f.h>
#include <libstats/distributions/gamma.h>
#include <libstats/distributions/gaussian.h>
#include <libstats/distributions/geometric.h>
#include <libstats/distributions/gumbel.h>
#include <libstats/distributions/half_normal.h>
#include <libstats/distributions/inverse_gamma.h>
#include <libstats/distributions/laplace.h>
#include <libstats/distributions/logistic.h>
#include <libstats/distributions/lognormal.h>
#include <libstats/distributions/negative_binomial.h>
#include <libstats/distributions/pareto.h>
#include <libstats/distributions/poisson.h>
#include <libstats/distributions/rayleigh.h>
#include <libstats/distributions/student_t.h>
#include <libstats/distributions/truncated_normal.h>
#include <libstats/distributions/uniform.h>
#include <libstats/distributions/von_mises.h>
#include <libstats/distributions/weibull.h>

#include "_common.h"

namespace nb = nanobind;
using namespace stats;

// ---------------------------------------------------------------------------
// Helper: construct via Result<T>::create(), raise ValueError on failure
// ---------------------------------------------------------------------------

template <typename Dist, typename... Args>
void checked_init(Dist* self, Args... args) {
    auto result = Dist::create(args...);
    if (result.isError()) {
        throw nb::value_error(result.message().c_str());
    }
    new (self) Dist(std::move(result).unwrap());
}

// ===========================================================================
NB_MODULE(_core, m) {
    m.doc() = "pylibstats native extension — SIMD-accelerated statistical distributions";

    // -----------------------------------------------------------------------
    // Gaussian (Normal)
    // -----------------------------------------------------------------------
    auto gaussian = nb::class_<GaussianDistribution>(m, "Gaussian",
        "Gaussian (normal) distribution N(mu, sigma).")
        .def("__init__",
             [](GaussianDistribution* self, double mu, double sigma) {
                 checked_init(self, mu, sigma);
             },
             nb::arg("mu") = 0.0, nb::arg("sigma") = 1.0)
        .def_prop_rw("mu",
            [](const GaussianDistribution& d) { return d.getMean(); },
            [](GaussianDistribution& d, double v) { d.setMean(v); },
            "Mean parameter μ.")
        .def_prop_rw("sigma",
            [](const GaussianDistribution& d) { return d.getStandardDeviation(); },
            [](GaussianDistribution& d, double v) { d.setStandardDeviation(v); },
            "Standard deviation parameter σ.");
    bind_common_methods<GaussianDistribution>(gaussian);

    // -----------------------------------------------------------------------
    // Exponential
    // -----------------------------------------------------------------------
    auto exponential = nb::class_<ExponentialDistribution>(m, "Exponential",
        "Exponential distribution Exp(lambda).")
        .def("__init__",
             [](ExponentialDistribution* self, double lam) {
                 checked_init(self, lam);
             },
             nb::arg("lam") = 1.0)
        .def_prop_rw("lam",
            [](const ExponentialDistribution& d) { return d.getLambda(); },
            [](ExponentialDistribution& d, double v) { d.setLambda(v); },
            "Rate parameter λ.");
    bind_common_methods<ExponentialDistribution>(exponential);

    // -----------------------------------------------------------------------
    // Uniform
    // -----------------------------------------------------------------------
    auto uniform = nb::class_<UniformDistribution>(m, "Uniform",
        "Continuous uniform distribution U(a, b).")
        .def("__init__",
             [](UniformDistribution* self, double a, double b) {
                 checked_init(self, a, b);
             },
             nb::arg("a") = 0.0, nb::arg("b") = 1.0)
        .def_prop_rw("a",
            [](const UniformDistribution& d) { return d.getLowerBound(); },
            [](UniformDistribution& d, double v) { d.setLowerBound(v); },
            "Lower bound a.")
        .def_prop_rw("b",
            [](const UniformDistribution& d) { return d.getUpperBound(); },
            [](UniformDistribution& d, double v) { d.setUpperBound(v); },
            "Upper bound b.");
    bind_common_methods<UniformDistribution>(uniform);

    // -----------------------------------------------------------------------
    // Poisson
    // -----------------------------------------------------------------------
    auto poisson = nb::class_<PoissonDistribution>(m, "Poisson",
        "Poisson distribution Pois(lambda).")
        .def("__init__",
             [](PoissonDistribution* self, double lam) {
                 checked_init(self, lam);
             },
             nb::arg("lam") = 1.0)
        .def_prop_rw("lam",
            [](const PoissonDistribution& d) { return d.getLambda(); },
            [](PoissonDistribution& d, double v) { d.setLambda(v); },
            "Rate parameter λ.");
    bind_common_methods<PoissonDistribution>(poisson);

    // -----------------------------------------------------------------------
    // Discrete Uniform
    // -----------------------------------------------------------------------
    auto discrete = nb::class_<DiscreteDistribution>(m, "DiscreteUniform",
        "Discrete uniform distribution over integers {a, a+1, ..., b}.")
        .def("__init__",
             [](DiscreteDistribution* self, int a, int b) {
                 checked_init(self, a, b);
             },
             nb::arg("a") = 0, nb::arg("b") = 1)
        .def_prop_rw("a",
            [](const DiscreteDistribution& d) { return d.getLowerBound(); },
            [](DiscreteDistribution& d, int v) { d.setLowerBound(v); },
            "Lower bound a (integer).")
        .def_prop_rw("b",
            [](const DiscreteDistribution& d) { return d.getUpperBound(); },
            [](DiscreteDistribution& d, int v) { d.setUpperBound(v); },
            "Upper bound b (integer).");
    bind_common_methods<DiscreteDistribution>(discrete);

    // -----------------------------------------------------------------------
    // Gamma
    // -----------------------------------------------------------------------
    auto gamma = nb::class_<GammaDistribution>(m, "Gamma",
        "Gamma distribution Gamma(alpha, beta).")
        .def("__init__",
             [](GammaDistribution* self, double alpha, double beta) {
                 checked_init(self, alpha, beta);
             },
             nb::arg("alpha") = 1.0, nb::arg("beta") = 1.0)
        .def_prop_rw("alpha",
            [](const GammaDistribution& d) { return d.getAlpha(); },
            [](GammaDistribution& d, double v) { d.setAlpha(v); },
            "Shape parameter α.")
        .def_prop_rw("beta",
            [](const GammaDistribution& d) { return d.getBeta(); },
            [](GammaDistribution& d, double v) { d.setBeta(v); },
            "Rate parameter β.");
    bind_common_methods<GammaDistribution>(gamma);

    // -----------------------------------------------------------------------
    // Beta
    // -----------------------------------------------------------------------
    auto beta = nb::class_<BetaDistribution>(m, "Beta",
        "Beta distribution Beta(alpha, beta).")
        .def("__init__",
             [](BetaDistribution* self, double alpha, double beta) {
                 checked_init(self, alpha, beta);
             },
             nb::arg("alpha") = 1.0, nb::arg("beta") = 1.0)
        .def_prop_rw("alpha",
            [](const BetaDistribution& d) { return d.getAlpha(); },
            [](BetaDistribution& d, double v) { d.setAlpha(v); },
            "Shape parameter α.")
        .def_prop_rw("beta",
            [](const BetaDistribution& d) { return d.getBeta(); },
            [](BetaDistribution& d, double v) { d.setBeta(v); },
            "Shape parameter β.");
    bind_common_methods<BetaDistribution>(beta);

    // -----------------------------------------------------------------------
    // Chi-Squared
    // -----------------------------------------------------------------------
    auto chi2 = nb::class_<ChiSquaredDistribution>(m, "ChiSquared",
        "Chi-squared distribution χ²(k).")
        .def("__init__",
             [](ChiSquaredDistribution* self, double k) {
                 checked_init(self, k);
             },
             nb::arg("k") = 1.0)
        .def_prop_rw("k",
            [](const ChiSquaredDistribution& d) { return d.getK(); },
            [](ChiSquaredDistribution& d, double v) { d.setK(v); },
            "Degrees of freedom k.");
    bind_common_methods<ChiSquaredDistribution>(chi2);

    // -----------------------------------------------------------------------
    // Student's t
    // -----------------------------------------------------------------------
    auto student_t = nb::class_<StudentTDistribution>(m, "StudentT",
        "Student's t distribution t(nu).")
        .def("__init__",
             [](StudentTDistribution* self, double nu) {
                 checked_init(self, nu);
             },
             nb::arg("nu") = 1.0)
        .def_prop_rw("nu",
            [](const StudentTDistribution& d) { return d.getNu(); },
            [](StudentTDistribution& d, double v) { d.setNu(v); },
            "Degrees of freedom ν.");
    bind_common_methods<StudentTDistribution>(student_t);

    // -----------------------------------------------------------------------
    // Log-Normal
    // -----------------------------------------------------------------------
    auto lognormal = nb::class_<LogNormalDistribution>(m, "LogNormal",
        "Log-normal distribution LogN(mu, sigma).")
        .def("__init__",
             [](LogNormalDistribution* self, double mu, double sigma) {
                 checked_init(self, mu, sigma);
             },
             nb::arg("mu") = 0.0, nb::arg("sigma") = 1.0)
        .def_prop_rw("mu",
            [](const LogNormalDistribution& d) { return d.getMu(); },
            [](LogNormalDistribution& d, double v) { d.setMu(v); },
            "Location parameter μ (log-mean).")
        .def_prop_rw("sigma",
            [](const LogNormalDistribution& d) { return d.getSigma(); },
            [](LogNormalDistribution& d, double v) { d.setSigma(v); },
            "Scale parameter σ (log-stddev, must be positive).");
    bind_common_methods<LogNormalDistribution>(lognormal);

    // -----------------------------------------------------------------------
    // Pareto
    // -----------------------------------------------------------------------
    auto pareto = nb::class_<ParetoDistribution>(m, "Pareto",
        "Pareto distribution Pareto(scale, alpha).")
        .def("__init__",
             [](ParetoDistribution* self, double scale, double alpha) {
                 checked_init(self, scale, alpha);
             },
             nb::arg("scale") = 1.0, nb::arg("alpha") = 1.0)
        .def_prop_rw("scale",
            [](const ParetoDistribution& d) { return d.getScale(); },
            [](ParetoDistribution& d, double v) { d.setScale(v); },
            "Minimum value (scale parameter x_m, must be positive).")
        .def_prop_rw("alpha",
            [](const ParetoDistribution& d) { return d.getAlpha(); },
            [](ParetoDistribution& d, double v) { d.setAlpha(v); },
            "Shape parameter α (tail index, must be positive).");
    bind_common_methods<ParetoDistribution>(pareto);

    // -----------------------------------------------------------------------
    // Weibull
    // -----------------------------------------------------------------------
    auto weibull = nb::class_<WeibullDistribution>(m, "Weibull",
        "Weibull distribution W(shape, scale).")
        .def("__init__",
             [](WeibullDistribution* self, double shape, double scale) {
                 checked_init(self, shape, scale);
             },
             nb::arg("shape") = 1.0, nb::arg("scale") = 1.0)
        .def_prop_rw("shape",
            [](const WeibullDistribution& d) { return d.getShape(); },
            [](WeibullDistribution& d, double v) { d.setShape(v); },
            "Shape parameter k (must be positive).")
        .def_prop_rw("scale",
            [](const WeibullDistribution& d) { return d.getScale(); },
            [](WeibullDistribution& d, double v) { d.setScale(v); },
            "Scale parameter λ (must be positive).");
    bind_common_methods<WeibullDistribution>(weibull);

    // -----------------------------------------------------------------------
    // Rayleigh
    // -----------------------------------------------------------------------
    auto rayleigh = nb::class_<RayleighDistribution>(m, "Rayleigh",
        "Rayleigh distribution R(sigma).")
        .def("__init__",
             [](RayleighDistribution* self, double sigma) {
                 checked_init(self, sigma);
             },
             nb::arg("sigma") = 1.0)
        .def_prop_rw("sigma",
            [](const RayleighDistribution& d) { return d.getSigma(); },
            [](RayleighDistribution& d, double v) { d.setSigma(v); },
            "Scale parameter σ (must be positive).");
    bind_common_methods<RayleighDistribution>(rayleigh);

    // -----------------------------------------------------------------------
    // Von Mises
    // -----------------------------------------------------------------------
    auto von_mises = nb::class_<VonMisesDistribution>(m, "VonMises",
        "Von Mises distribution VM(mu, kappa). Mu is wrapped to (-pi, pi].")
        .def("__init__",
             [](VonMisesDistribution* self, double mu, double kappa) {
                 checked_init(self, mu, kappa);
             },
             nb::arg("mu") = 0.0, nb::arg("kappa") = 1.0)
        .def_prop_rw("mu",
            [](const VonMisesDistribution& d) { return d.getMu(); },
            [](VonMisesDistribution& d, double v) { d.setMu(v); },
            "Mean direction μ (any finite real; stored wrapped to (-π, π]).")
        .def_prop_rw("kappa",
            [](const VonMisesDistribution& d) { return d.getKappa(); },
            [](VonMisesDistribution& d, double v) { d.setKappa(v); },
            "Concentration parameter κ (≥ 0; 0 = uniform).");
    bind_common_methods<VonMisesDistribution>(von_mises);

    // -----------------------------------------------------------------------
    // Binomial
    // -----------------------------------------------------------------------
    auto binomial = nb::class_<BinomialDistribution>(m, "Binomial",
        "Binomial distribution B(n, p).")
        .def("__init__",
             [](BinomialDistribution* self, int n, double p) {
                 checked_init(self, n, p);
             },
             nb::arg("n") = 10, nb::arg("p") = 0.5)
        .def_prop_rw("n",
            [](const BinomialDistribution& d) { return d.getN(); },
            [](BinomialDistribution& d, int v) { d.setN(v); },
            "Number of trials n (positive integer).")
        .def_prop_rw("p",
            [](const BinomialDistribution& d) { return d.getP(); },
            [](BinomialDistribution& d, double v) { d.setP(v); },
            "Success probability p (in [0, 1]).");
    bind_common_methods<BinomialDistribution>(binomial);

    // -----------------------------------------------------------------------
    // Negative Binomial
    // -----------------------------------------------------------------------
    auto neg_binomial = nb::class_<NegativeBinomialDistribution>(m, "NegativeBinomial",
        "Negative Binomial distribution NB(r, p). Real-valued r supported.")
        .def("__init__",
             [](NegativeBinomialDistribution* self, double r, double p) {
                 checked_init(self, r, p);
             },
             nb::arg("r") = 1.0, nb::arg("p") = 0.5)
        .def_prop_rw("r",
            [](const NegativeBinomialDistribution& d) { return d.getR(); },
            [](NegativeBinomialDistribution& d, double v) { d.setR(v); },
            "Number of successes r (positive, real-valued).")
        .def_prop_rw("p",
            [](const NegativeBinomialDistribution& d) { return d.getP(); },
            [](NegativeBinomialDistribution& d, double v) { d.setP(v); },
            "Success probability p (in (0, 1]).");
    bind_common_methods<NegativeBinomialDistribution>(neg_binomial);

    // -----------------------------------------------------------------------
    // Geometric
    // -----------------------------------------------------------------------
    auto geometric = nb::class_<GeometricDistribution>(m, "Geometric",
        "Geometric distribution Geo(p). Models failures before first success.")
        .def("__init__",
             [](GeometricDistribution* self, double p) {
                 checked_init(self, p);
             },
             nb::arg("p") = 0.5)
        .def_prop_rw("p",
            [](const GeometricDistribution& d) { return d.getP(); },
            [](GeometricDistribution& d, double v) { d.setP(v); },
            "Success probability p (in (0, 1]).");
    bind_common_methods<GeometricDistribution>(geometric);

    // -----------------------------------------------------------------------
    // Laplace
    // -----------------------------------------------------------------------
    auto laplace = nb::class_<LaplaceDistribution>(m, "Laplace",
        "Laplace (double-exponential) distribution Laplace(mu, b).")
        .def("__init__",
             [](LaplaceDistribution* self, double mu, double b) {
                 checked_init(self, mu, b);
             },
             nb::arg("mu") = 0.0, nb::arg("b") = 1.0)
        .def_prop_rw("mu",
            [](const LaplaceDistribution& d) { return d.getMu(); },
            [](LaplaceDistribution& d, double v) { d.setMu(v); },
            "Location parameter \u03bc (any finite value).")
        .def_prop_rw("b",
            [](const LaplaceDistribution& d) { return d.getB(); },
            [](LaplaceDistribution& d, double v) { d.setB(v); },
            "Scale parameter b (must be positive).");
    bind_common_methods<LaplaceDistribution>(laplace);

    // -----------------------------------------------------------------------
    // Cauchy
    // -----------------------------------------------------------------------
    auto cauchy = nb::class_<CauchyDistribution>(m, "Cauchy",
        "Cauchy distribution Cauchy(x0, gamma). Mean/variance/skewness/kurtosis are NaN.")
        .def("__init__",
             [](CauchyDistribution* self, double x0, double gamma) {
                 checked_init(self, x0, gamma);
             },
             nb::arg("x0") = 0.0, nb::arg("gamma") = 1.0)
        .def_prop_rw("x0",
            [](const CauchyDistribution& d) { return d.getX0(); },
            [](CauchyDistribution& d, double v) { d.setX0(v); },
            "Location parameter x\u2080 (any finite value).")
        .def_prop_rw("gamma",
            [](const CauchyDistribution& d) { return d.getGamma(); },
            [](CauchyDistribution& d, double v) { d.setGamma(v); },
            "Scale parameter \u03b3 (must be positive).");
    bind_common_methods<CauchyDistribution>(cauchy);

    // -----------------------------------------------------------------------
    // Logistic
    // -----------------------------------------------------------------------
    auto logistic = nb::class_<LogisticDistribution>(m, "Logistic",
        "Logistic distribution Logistic(mu, s).")
        .def("__init__",
             [](LogisticDistribution* self, double mu, double s) {
                 checked_init(self, mu, s);
             },
             nb::arg("mu") = 0.0, nb::arg("s") = 1.0)
        .def_prop_rw("mu",
            [](const LogisticDistribution& d) { return d.getMu(); },
            [](LogisticDistribution& d, double v) { d.setMu(v); },
            "Location parameter \u03bc (any finite value).")
        .def_prop_rw("s",
            [](const LogisticDistribution& d) { return d.getS(); },
            [](LogisticDistribution& d, double v) { d.setS(v); },
            "Scale parameter s (must be positive).");
    bind_common_methods<LogisticDistribution>(logistic);

    // -----------------------------------------------------------------------
    // Gumbel
    // -----------------------------------------------------------------------
    auto gumbel = nb::class_<GumbelDistribution>(m, "Gumbel",
        "Gumbel (maximum, right-skewed) distribution Gumbel(mu, beta).")
        .def("__init__",
             [](GumbelDistribution* self, double mu, double beta) {
                 checked_init(self, mu, beta);
             },
             nb::arg("mu") = 0.0, nb::arg("beta") = 1.0)
        .def_prop_rw("mu",
            [](const GumbelDistribution& d) { return d.getMu(); },
            [](GumbelDistribution& d, double v) { d.setMu(v); },
            "Location parameter \u03bc (any finite value).")
        .def_prop_rw("beta",
            [](const GumbelDistribution& d) { return d.getBeta(); },
            [](GumbelDistribution& d, double v) { d.setBeta(v); },
            "Scale parameter \u03b2 (must be positive).");
    bind_common_methods<GumbelDistribution>(gumbel);

    // -----------------------------------------------------------------------
    // Erlang
    // -----------------------------------------------------------------------
    auto erlang = nb::class_<ErlangDistribution>(m, "Erlang",
        "Erlang distribution Erlang(k, lambda). Gamma restricted to integer shape k.")
        .def("__init__",
             [](ErlangDistribution* self, int k, double lam) {
                 checked_init(self, k, lam);
             },
             nb::arg("k") = 1, nb::arg("lam") = 1.0)
        .def_prop_rw("k",
            [](const ErlangDistribution& d) { return d.getK(); },
            [](ErlangDistribution& d, int v) { d.setK(v); },
            "Shape parameter k (positive integer).")
        .def_prop_rw("lam",
            [](const ErlangDistribution& d) { return d.getLambda(); },
            [](ErlangDistribution& d, double v) { d.setLambda(v); },
            "Rate parameter \u03bb (must be positive).");
    bind_common_methods<ErlangDistribution>(erlang);

    // -----------------------------------------------------------------------
    // Fisher F
    // -----------------------------------------------------------------------
    auto fisher_f = nb::class_<FDistribution>(m, "FisherF",
        "Fisher-Snedecor F distribution F(d1, d2).")
        .def("__init__",
             [](FDistribution* self, double d1, double d2) {
                 checked_init(self, d1, d2);
             },
             nb::arg("d1") = 1.0, nb::arg("d2") = 1.0)
        .def_prop_rw("d1",
            [](const FDistribution& d) { return d.getD1(); },
            [](FDistribution& d, double v) { d.setD1(v); },
            "Numerator degrees of freedom d1 (must be positive).")
        .def_prop_rw("d2",
            [](const FDistribution& d) { return d.getD2(); },
            [](FDistribution& d, double v) { d.setD2(v); },
            "Denominator degrees of freedom d2 (must be positive).");
    bind_common_methods<FDistribution>(fisher_f);

    // -----------------------------------------------------------------------
    // Inverse Gamma
    // -----------------------------------------------------------------------
    auto inverse_gamma = nb::class_<InverseGammaDistribution>(m, "InverseGamma",
        "Inverse Gamma distribution InvGamma(alpha, beta). Beta is a SCALE parameter.")
        .def("__init__",
             [](InverseGammaDistribution* self, double alpha, double beta) {
                 checked_init(self, alpha, beta);
             },
             nb::arg("alpha") = 1.0, nb::arg("beta") = 1.0)
        .def_prop_rw("alpha",
            [](const InverseGammaDistribution& d) { return d.getAlpha(); },
            [](InverseGammaDistribution& d, double v) { d.setAlpha(v); },
            "Shape parameter \u03b1 (must be positive).")
        .def_prop_rw("beta",
            [](const InverseGammaDistribution& d) { return d.getBeta(); },
            [](InverseGammaDistribution& d, double v) { d.setBeta(v); },
            "Scale parameter \u03b2 (must be positive).");
    bind_common_methods<InverseGammaDistribution>(inverse_gamma);

    // -----------------------------------------------------------------------
    // Half-Normal
    // -----------------------------------------------------------------------
    auto half_normal = nb::class_<HalfNormalDistribution>(m, "HalfNormal",
        "Half-normal distribution HalfNormal(sigma).")
        .def("__init__",
             [](HalfNormalDistribution* self, double sigma) {
                 checked_init(self, sigma);
             },
             nb::arg("sigma") = 1.0)
        .def_prop_rw("sigma",
            [](const HalfNormalDistribution& d) { return d.getSigma(); },
            [](HalfNormalDistribution& d, double v) { d.setSigma(v); },
            "Scale parameter \u03c3 (must be positive).");
    bind_common_methods<HalfNormalDistribution>(half_normal);

    // -----------------------------------------------------------------------
    // Bernoulli
    // -----------------------------------------------------------------------
    auto bernoulli = nb::class_<BernoulliDistribution>(m, "Bernoulli",
        "Bernoulli distribution Bernoulli(p). p in [0, 1] inclusive.")
        .def("__init__",
             [](BernoulliDistribution* self, double p) {
                 checked_init(self, p);
             },
             nb::arg("p") = 0.5)
        .def_prop_rw("p",
            [](const BernoulliDistribution& d) { return d.getP(); },
            [](BernoulliDistribution& d, double v) { d.setP(v); },
            "Success probability p (in [0, 1] inclusive).");
    bind_common_methods<BernoulliDistribution>(bernoulli);

    // -----------------------------------------------------------------------
    // Truncated Normal
    // -----------------------------------------------------------------------
    auto truncated_normal = nb::class_<TruncatedNormalDistribution>(m, "TruncatedNormal",
        "Truncated normal distribution N(mu, sigma) restricted to [a, b]. "
        "Bounds are absolute coordinates (not standardized as in scipy) and "
        "may be +/-inf; windows whose normalization underflows double "
        "(roughly beyond +/-37.5 sigma) are rejected.")
        .def("__init__",
             [](TruncatedNormalDistribution* self, double mu, double sigma, double a, double b) {
                 checked_init(self, mu, sigma, a, b);
             },
             nb::arg("mu") = 0.0, nb::arg("sigma") = 1.0,
             nb::arg("a") = -std::numeric_limits<double>::infinity(),
             nb::arg("b") = std::numeric_limits<double>::infinity())
        .def_prop_rw("mu",
            [](const TruncatedNormalDistribution& d) { return d.getMu(); },
            [](TruncatedNormalDistribution& d, double v) { d.setMu(v); },
            "Location parameter \u03bc of the parent normal (any finite value).")
        .def_prop_rw("sigma",
            [](const TruncatedNormalDistribution& d) { return d.getSigma(); },
            [](TruncatedNormalDistribution& d, double v) { d.setSigma(v); },
            "Scale parameter \u03c3 of the parent normal (must be positive).")
        .def_prop_rw("a",
            [](const TruncatedNormalDistribution& d) { return d.getLowerBound(); },
            [](TruncatedNormalDistribution& d, double v) { d.setLowerBound(v); },
            "Lower truncation bound a (absolute; -inf for untruncated).")
        .def_prop_rw("b",
            [](const TruncatedNormalDistribution& d) { return d.getUpperBound(); },
            [](TruncatedNormalDistribution& d, double v) { d.setUpperBound(v); },
            "Upper truncation bound b (absolute; +inf for untruncated).");
    bind_common_methods<TruncatedNormalDistribution>(truncated_normal);
}
