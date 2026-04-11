#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <random>
#include <vector>

namespace nb = nanobind;

/// NumPy array type alias for contiguous float64 input
using NpArrayIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// ---------------------------------------------------------------------------
// Helper: convert std::vector<double> → owned NumPy array (copies data once)
// ---------------------------------------------------------------------------

inline nb::object vec_to_numpy(const std::vector<double>& vec) {
    const size_t n = vec.size();
    // Wrap raw pointer with nullptr owner → nanobind copies on return
    return nb::ndarray<nb::numpy, const double, nb::ndim<1>>(
               vec.data(), {n}, nb::handle()).cast();
}

inline nb::object vec_to_numpy(std::vector<double>&& vec) {
    // Move into a heap-allocated vector so the capsule can own it
    auto* owned = new std::vector<double>(std::move(vec));
    nb::capsule deleter(owned, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    const size_t n = owned->size();
    return nb::cast(
        nb::ndarray<nb::numpy, double, nb::ndim<1>>(
            owned->data(), {n}, deleter),
        nb::rv_policy::move);
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
    // Uses getBatchProbabilities / getBatchLogProbabilities / getBatchCumulativeProbabilities
    // from DistributionBase, which auto-dispatch to SIMD/parallel internally.
    cls.def("pdf",
            [](const Dist& d, NpArrayIn x) -> nb::object {
                std::vector<double> input(x.data(), x.data() + x.shape(0));
                std::vector<double> result;
                {
                    nb::gil_scoped_release release;
                    result = d.getBatchProbabilities(input);
                }
                return vec_to_numpy(std::move(result));
            },
            nb::arg("x").noconvert(),
            "Batch PDF: accepts a 1-D float64 NumPy array, returns array of densities.")

       .def("log_pdf",
            [](const Dist& d, NpArrayIn x) -> nb::object {
                std::vector<double> input(x.data(), x.data() + x.shape(0));
                std::vector<double> result;
                {
                    nb::gil_scoped_release release;
                    result = d.getBatchLogProbabilities(input);
                }
                return vec_to_numpy(std::move(result));
            },
            nb::arg("x").noconvert(),
            "Batch log-PDF: accepts a 1-D float64 NumPy array.")

       .def("cdf",
            [](const Dist& d, NpArrayIn x) -> nb::object {
                std::vector<double> input(x.data(), x.data() + x.shape(0));
                std::vector<double> result;
                {
                    nb::gil_scoped_release release;
                    result = d.getBatchCumulativeProbabilities(input);
                }
                return vec_to_numpy(std::move(result));
            },
            nb::arg("x").noconvert(),
            "Batch CDF: accepts a 1-D float64 NumPy array.");

    // -- Fitting --------------------------------------------------------------
    cls.def("fit", [](Dist& d, NpArrayIn data) {
                std::vector<double> vec(data.data(), data.data() + data.shape(0));
                d.fit(vec);
            },
            nb::arg("data"),
            "Fit distribution parameters to data via maximum likelihood estimation.");

    // -- Sampling -------------------------------------------------------------
    cls.def("sample",
            [](const Dist& d, size_t n, nb::object seed) -> nb::object {
                std::mt19937 rng;
                if (seed.is_none()) {
                    rng.seed(std::random_device{}());
                } else {
                    rng.seed(nb::cast<unsigned int>(seed));
                }
                auto samples = d.sample(rng, n);
                return vec_to_numpy(std::move(samples));
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
