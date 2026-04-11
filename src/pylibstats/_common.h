#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <random>
#include <span>
#include <vector>

namespace nb = nanobind;

/// NumPy array type aliases for nanobind
using NpArrayIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NpArrayOut = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NpArrayOwned = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// ---------------------------------------------------------------------------
// Batch helper: allocate output array + call span-based method with GIL released
// ---------------------------------------------------------------------------

/// Generic batch wrapper that calls a member taking (span<const double>, span<double>).
/// The member pointer is passed as a template parameter to avoid std::function overhead.
template <typename Dist,
          void (Dist::*Method)(std::span<const double>, std::span<double>,
                               const detail::PerformanceHint&) const>
NpArrayOwned batch_call(const Dist& dist, NpArrayIn input) {
    const size_t n = input.shape(0);
    auto result = nb::ndarray<nb::numpy, double, nb::ndim<1>>(/* shape = */ {n});
    double* out_ptr = result.data();
    const double* in_ptr = input.data();
    {
        nb::gil_scoped_release release;
        (dist.*Method)(std::span<const double>{in_ptr, n},
                       std::span<double>{out_ptr, n}, {});
    }
    return result;
}

// ---------------------------------------------------------------------------
// bind_common_methods — shared across all distribution classes
// ---------------------------------------------------------------------------

template <typename Dist, typename PyClass>
void bind_common_methods(PyClass& cls) {
    // -- Scalar probability functions (preferred first in overload order) ------
    cls.def("pdf", [](const Dist& d, double x) { return d.getProbability(x); },
            nb::arg("x"), "Evaluate probability density/mass at x.")

       .def("log_pdf", [](const Dist& d, double x) { return d.getLogProbability(x); },
            nb::arg("x"), "Evaluate log probability density/mass at x.")

       .def("cdf", [](const Dist& d, double x) { return d.getCumulativeProbability(x); },
            nb::arg("x"), "Evaluate cumulative distribution function at x.")

       .def("ppf", [](const Dist& d, double p) { return d.getQuantile(p); },
            nb::arg("p"), "Quantile function (inverse CDF). Returns x such that P(X <= x) = p.");

    // -- Batch probability functions (NumPy arrays, GIL-releasing) ------------
    cls.def("pdf",
            &batch_call<Dist, &Dist::getProbability>,
            nb::arg("x").noconvert(),
            "Batch PDF: accepts a 1-D float64 NumPy array, returns array of densities.")

       .def("log_pdf",
            &batch_call<Dist, &Dist::getLogProbability>,
            nb::arg("x").noconvert(),
            "Batch log-PDF: accepts a 1-D float64 NumPy array.")

       .def("cdf",
            &batch_call<Dist, &Dist::getCumulativeProbability>,
            nb::arg("x").noconvert(),
            "Batch CDF: accepts a 1-D float64 NumPy array.");

    // -- Fitting --------------------------------------------------------------
    cls.def("fit", [](Dist& d, nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> data) {
                std::vector<double> vec(data.data(), data.data() + data.shape(0));
                d.fit(vec);
            },
            nb::arg("data"),
            "Fit distribution parameters to data via maximum likelihood estimation.");

    // -- Sampling -------------------------------------------------------------
    cls.def("sample",
            [](const Dist& d, size_t n, nb::object seed) -> NpArrayOwned {
                std::mt19937 rng;
                if (seed.is_none()) {
                    rng.seed(std::random_device{}());
                } else {
                    rng.seed(nb::cast<unsigned int>(seed));
                }
                auto samples = d.sample(rng, n);
                auto result = nb::ndarray<nb::numpy, double, nb::ndim<1>>({n});
                std::copy(samples.begin(), samples.end(), result.data());
                return result;
            },
            nb::arg("n") = 1, nb::arg("seed") = nb::none(),
            "Generate random samples. Returns a 1-D NumPy array.");

    // -- Read-only moment properties ------------------------------------------
    cls.def_prop_ro("mean", [](const Dist& d) { return d.getMean(); },
                    "Distribution mean (first moment).")
       .def_prop_ro("variance", [](const Dist& d) { return d.getVariance(); },
                    "Distribution variance (second central moment).")
       .def_prop_ro("std", [](const Dist& d) { return std::sqrt(d.getVariance()); },
                    "Standard deviation (sqrt of variance).")
       .def_prop_ro("skewness", [](const Dist& d) { return d.getSkewness(); },
                    "Distribution skewness (third standardised moment).")
       .def_prop_ro("kurtosis", [](const Dist& d) { return d.getKurtosis(); },
                    "Distribution excess kurtosis (fourth standardised moment).");

    // -- Metadata properties --------------------------------------------------
    cls.def_prop_ro("name", [](const Dist& d) { return d.getDistributionName(); })
       .def_prop_ro("support",
                    [](const Dist& d) {
                        return nb::make_tuple(d.getSupportLowerBound(),
                                              d.getSupportUpperBound());
                    },
                    "Support interval as (lower, upper) tuple.")
       .def_prop_ro("is_discrete", [](const Dist& d) { return d.isDiscrete(); })
       .def_prop_ro("num_parameters", [](const Dist& d) { return d.getNumParameters(); });

    // -- String representation ------------------------------------------------
    cls.def("__repr__", [](const Dist& d) { return d.toString(); });
}
