"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop

def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


def test_rejects_negative_rate():
    with pytest.raises(ValueError):
         simulate(N0 = 1000,lam = -0.4)
        #with pytest.raise we did it

def test_matches_analytical_law():
    N0,lam = 10000,0.4
    avg = np.mean([simulate(N0,lam) for _ in range(1000)],axis = 0)
    t = np.arange(len(avg)) * 0.05
    expected = N0 * np.exp(-lam * t)
    assert avg == pytest.approx(expected, rel=0.1)
#   with pytest.approx(expected) we did it
