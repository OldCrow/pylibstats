/// pylibstats native extension module.
///
/// Binds all libstats distribution classes to Python via nanobind, exposing
/// scalar and SIMD-accelerated batch operations through NumPy arrays.

#include <nanobind/nanobind.h>

// libstats distribution headers
#include <libstats/distributions/beta.h>
#include <libstats/distributions/chi_squared.h>
#include <libstats/distributions/discrete.h>
#include <libstats/distributions/exponential.h>
#include <libstats/distributions/gamma.h>
#include <libstats/distributions/gaussian.h>
#include <libstats/distributions/poisson.h>
#include <libstats/distributions/student_t.h>
#include <libstats/distributions/uniform.h>

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
}
