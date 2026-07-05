#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace nb = nanobind;

/// NumPy array type alias for contiguous float64 input
using NpArrayIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// ---------------------------------------------------------------------------
// Helpers: wrap raw buffers into owned NumPy arrays
// ---------------------------------------------------------------------------

/// Wrap a new[]-allocated buffer into a NumPy array with capsule ownership.
/// The buffer must already be filled; caller relinquishes ownership.
inline nb::object buf_to_numpy(double* buf, size_t n) {
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, {n}, owner),
        nb::rv_policy::move);
}

/// Convert std::vector<double> to an owned NumPy array (for sampling/fit).
inline nb::object vec_to_numpy(std::vector<double>&& vec) {
    auto* owned = new std::vector<double>(std::move(vec));
    nb::capsule deleter(owned, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    const size_t n = owned->size();
    return nb::cast(
        nb::ndarray<nb::numpy, double, nb::ndim<1>>(owned->data(), {n}, deleter),
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

    // -- Batch probability functions (zero-copy span path, GIL-releasing) -----
    // Passes NumPy buffer pointers directly to the span-based batch methods,
    // which auto-dispatch to SIMD/parallel internally. No intermediate vector.
    cls.def("pdf",
            [](const Dist& d, NpArrayIn x) -> nb::object {
                const size_t n = x.shape(0);
                auto* buf = new double[n];
                {
                    nb::gil_scoped_release release;
                    d.getProbability(
                        std::span<const double>{x.data(), n},
                        std::span<double>{buf, n});
                }
                return buf_to_numpy(buf, n);
            },
            nb::arg("x").noconvert(),
            "Batch PDF: accepts a 1-D float64 NumPy array, returns array of densities.")

       .def("log_pdf",
            [](const Dist& d, NpArrayIn x) -> nb::object {
                const size_t n = x.shape(0);
                auto* buf = new double[n];
                {
                    nb::gil_scoped_release release;
                    d.getLogProbability(
                        std::span<const double>{x.data(), n},
                        std::span<double>{buf, n});
                }
                return buf_to_numpy(buf, n);
            },
            nb::arg("x").noconvert(),
            "Batch log-PDF: accepts a 1-D float64 NumPy array.")

       .def("cdf",
            [](const Dist& d, NpArrayIn x) -> nb::object {
                const size_t n = x.shape(0);
                auto* buf = new double[n];
                {
                    nb::gil_scoped_release release;
                    d.getCumulativeProbability(
                        std::span<const double>{x.data(), n},
                        std::span<double>{buf, n});
                }
                return buf_to_numpy(buf, n);
            },
            nb::arg("x").noconvert(),
            "Batch CDF: accepts a 1-D float64 NumPy array.")

       .def("ppf",
            [](const Dist& d, NpArrayIn p) -> nb::object {
                const size_t n = p.shape(0);
                auto* buf = new double[n];
                {
                    nb::gil_scoped_release release;
                    const double* values = p.data();
                    for (size_t i = 0; i < n; ++i) {
                        buf[i] = d.getQuantile(values[i]);
                    }
                }
                return buf_to_numpy(buf, n);
            },
            nb::arg("p").noconvert(),
            "Batch PPF/quantile: accepts a 1-D float64 NumPy array.");

    // -- Fitting --------------------------------------------------------------
    cls.def("fit", [](Dist& d, NpArrayIn data) {
                std::vector<double> vec(data.data(), data.data() + data.shape(0));
                {
                    nb::gil_scoped_release release;
                    d.fit(vec);
                }
            },
            nb::arg("data"),
            "Fit distribution parameters to data via maximum likelihood estimation.");

    // -- Sampling -------------------------------------------------------------
    // Uses a thread_local mt19937 for unseeded calls so that two rapid unseeded
    // calls on the same thread advance the same engine state rather than each
    // seeding independently from random_device (which can return the same value
    // within the same clock tick, producing correlated samples).
    cls.def("sample",
            [](const Dist& d, size_t n, nb::object seed) -> nb::object {
                if (seed.is_none()) {
                    // Thread-local engine seeded once per thread at first use.
                    static thread_local std::mt19937 tls_rng{std::random_device{}()};
                    std::vector<double> samples;
                    {
                        nb::gil_scoped_release release;
                        samples = d.sample(tls_rng, n);
                    }
                    return vec_to_numpy(std::move(samples));
                } else {
                    const auto raw_seed = nb::cast<std::uint64_t>(seed);
                    std::seed_seq seed_seq{
                        static_cast<std::uint32_t>(raw_seed),
                        static_cast<std::uint32_t>(raw_seed >> 32)};
                    std::mt19937 rng{seed_seq};
                    std::vector<double> samples;
                    {
                        nb::gil_scoped_release release;
                        samples = d.sample(rng, n);
                    }
                    return vec_to_numpy(std::move(samples));
                }
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
