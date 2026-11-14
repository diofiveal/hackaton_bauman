"""Numerical kernel tests against SciPy and oscillator state invariants."""

import unittest

import numpy as np
from scipy.signal import lfilter

from coefficients import calculate_smallwood_coefficients
from oscillator import (
    SmallwoodState,
    calculate_one_oscillator,
    calculate_one_oscillator_maximax,
)
from smallwood import calculate_srs_1d


class OscillatorTest(unittest.TestCase):
    def setUp(self):
        self.coefficients = calculate_smallwood_coefficients(350.0, 6660.0, 10.0)

    def test_responses_against_lfilter(self):
        signals = {
            "zero": np.zeros(120),
            "constant": np.ones(120),
            "impulse": np.r_[1.0, np.zeros(119)],
            "negative_impulse": np.r_[-1.0, np.zeros(119)],
            "half_sine": np.r_[np.sin(np.linspace(0, np.pi, 40)), np.zeros(80)],
            "changing_sign": np.tile([1.0, -1.0, 0.0], 40),
        }
        c = self.coefficients
        for name, signal in signals.items():
            with self.subTest(signal=name):
                expected = lfilter([c.b0, c.b1, c.b2], [1.0, c.a1, c.a2], signal)
                actual = calculate_one_oscillator(signal, c)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-11)
                self.assertEqual(
                    calculate_one_oscillator_maximax(signal, c), max(abs(actual))
                )

    def test_chunked_state_and_reset(self):
        signal = np.sin(np.arange(120) * 0.2)
        c = self.coefficients
        state = SmallwoodState()
        first = calculate_one_oscillator(signal[:60], c, state=state)
        second = calculate_one_oscillator(
            signal[60:], c, state=state, reset_state=False
        )
        expected = calculate_one_oscillator(signal, c)
        np.testing.assert_array_equal(np.r_[first, second], expected)
        np.testing.assert_array_equal(
            calculate_one_oscillator(signal, c, state=state), expected
        )
        state.reset()
        self.assertEqual(state.peak_abs_g, 0.0)

    def test_zero_tail_length_and_peak(self):
        c = self.coefficients
        signal = np.r_[1.0, np.zeros(9)]
        response = calculate_one_oscillator(
            signal, c, tail_mode="zero_tail", tail_time_constants=2.0
        )
        count = int(
            np.ceil(
                2.0 * c.quality_factor_q / (np.pi * c.frequency_hz) * c.sample_rate_hz
            )
        )
        self.assertEqual(response.size, signal.size + count)
        self.assertGreaterEqual(
            max(abs(response)), max(abs(calculate_one_oscillator(signal, c)))
        )

    def test_1d_grid_uses_the_same_kernel(self):
        signal = np.r_[1.0, np.zeros(99)]
        frequencies = [10.0, 100.0, 1000.0]
        expected = [
            calculate_one_oscillator_maximax(
                signal, calculate_smallwood_coefficients(f, 6660.0, 10.0)
            )
            for f in frequencies
        ]
        np.testing.assert_array_equal(
            calculate_srs_1d(signal, frequencies, 6660.0, 10.0), expected
        )

    def test_invalid_signal_and_coefficients(self):
        for signal in ([], [np.nan], [np.inf], [[1, 2]]):
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                calculate_one_oscillator(signal, self.coefficients)
        with self.assertRaises(ValueError):
            calculate_smallwood_coefficients(100, 6660, 0.5)
