/// pylibstats native extension module.
///
/// Binds all libstats distribution classes to Python via nanobind, exposing
/// scalar and SIMD-accelerated batch operations through NumPy arrays.

#include <nanobind/nanobind.h>

// libstats distribution headers
#include <libstats/distributions/beta.h>
#include <libstats/distributions/binomial.h>
#include <libstats/distributions/chi_squared.h>
#include <libstats/distributions/discrete.h>
#include <libstats/distributions/exponential.h>
#include <libstats/distributions/gamma.h>
#include <libstats/distributions/gaussian.h>
#include <libstats/distributions/lognormal.h>
#include <libstats/distributions/negative_binomial.h>
#include <libstats/distributions/pareto.h>
#include <libstats/distributions/poisson.h>
#include <libstats/distributions/rayleigh.h>
#include <libstats/distributions/student_t.h>
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
        throw nb::value_error(result.message.c_str());
    }
    new (self) Dist(std::move(result.value));
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
}
