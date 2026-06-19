"""Tests for pylibstats.Binomial (BinomialDistribution bindings)."""

import numpy as np
import pytest

import pylibstats


class TestBinomialConstruction:
    def test_default_params(self):
        d = pylibstats.Binomial()
        assert d.n == 10
        assert d.p == pytest.approx(0.5)

    def test_custom_params(self):
        d = pylibstats.Binomial(n=20, p=0.3)
        assert d.n == 20
        assert d.p == pytest.approx(0.3)

    def test_n_is_int(self, binomial):
        assert isinstance(binomial.n, int)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Binomial(0, 0.5)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Binomial(-1, 0.5)

    def test_float_n_raises(self):
        with pytest.raises((ValueError, TypeError)):
            pylibstats.Binomial(10.5, 0.5)  # type: ignore[arg-type]

    def test_numpy_int64_accepted(self):
        # S-2: np.int64 (and other NumPy integer scalars) must be accepted.
        import numpy as np
        n = np.int64(20)
        d = pylibstats.Binomial(n, 0.4)
        assert d.n == 20

    def test_numpy_int_setter_accepted(self):
        # S-2: n setter must accept NumPy integer scalars.
        import numpy as np
        d = pylibstats.Binomial(10, 0.5)
        d.n = np.int32(15)
        assert d.n == 15

    def test_p_above_one_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Binomial(10, 1.1)

    def test_p_negative_raises(self):
        with pytest.raises(ValueError):
            pylibstats.Binomial(10, -0.1)

    def test_p_zero_valid(self):
        d = pylibstats.Binomial(10, 0.0)
        assert d.p == pytest.approx(0.0)

    def test_p_one_valid(self):
        d = pylibstats.Binomial(10, 1.0)
        assert d.p == pytest.approx(1.0)


class TestBinomialScalar:
    def test_pmf_known_value(self, binomial):
        # B(10, 0.5): PMF(5) = C(10,5)/2^10 = 252/1024
        assert binomial.pdf(5.0) == pytest.approx(252.0 / 1024.0, rel=1e-10)

    def test_pmf_out_of_range_is_zero(self, binomial):
        assert binomial.pdf(-1.0) == pytest.approx(0.0, abs=1e-12)
        assert binomial.pdf(11.0) == pytest.approx(0.0, abs=1e-12)

    def test_cdf_known_value(self, binomial):
        # B(10, 0.5): CDF(5) = 0.623046875
        assert binomial.cdf(5.0) == pytest.approx(0.623046875, rel=1e-6)

    def test_cdf_at_n_is_one(self, binomial):
        assert binomial.cdf(10.0) == pytest.approx(1.0, abs=1e-10)

    def test_log_pdf_consistency(self, binomial):
        import math
        pmf5 = binomial.pdf(5.0)
        assert binomial.log_pdf(5.0) == pytest.approx(math.log(pmf5), rel=1e-10)

    def test_ppf_round_trip(self, binomial):
        cdf5 = binomial.cdf(5.0)
        assert binomial.ppf(cdf5) == pytest.approx(5.0, abs=0.5)


class TestBinomialBatch:
    def test_batch_shape(self, binomial):
        x = np.arange(0, 11, dtype=float)
        assert binomial.pdf(x).shape == (11,)

    def test_batch_matches_scalar(self, binomial):
        x = np.array([0.0, 3.0, 5.0, 8.0, 10.0])
        np.testing.assert_allclose(binomial.pdf(x),
                                   [binomial.pdf(v) for v in x], rtol=1e-12)

    def test_batch_pmf_sums_to_one(self, binomial):
        x = np.arange(0, 11, dtype=float)
        assert binomial.pdf(x).sum() == pytest.approx(1.0, rel=1e-10)


class TestBinomialProperties:
    def test_mean(self, binomial):
        assert binomial.mean == pytest.approx(5.0, rel=1e-10)

    def test_variance(self, binomial):
        assert binomial.variance == pytest.approx(2.5, rel=1e-10)

    def test_is_discrete(self, binomial):
        assert binomial.is_discrete

    def test_support(self, binomial):
        lower, upper = binomial.support
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(10.0)


class TestBinomialFitAndSample:
    def test_sample_in_range(self, binomial):
        s = binomial.sample(n=500, seed=12)
        assert np.all(s >= 0) and np.all(s <= 10)

    def test_fit_recovers_p(self):
        d = pylibstats.Binomial(10, 0.7)
        data = d.sample(n=10_000, seed=13)
        fitted = pylibstats.Binomial(10, 0.5)
        fitted.fit(data)
        assert fitted.p == pytest.approx(0.7, abs=0.05)
