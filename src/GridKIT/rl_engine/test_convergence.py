# rl_engine/test_convergence.py
"""Unit tests for plateau-based convergence detection + batch scaling.

The convergence tracker is pure logic (no Ray/RLlib in the class under test),
the scaling tests only touch Trainer's pure factor/batch helpers (no Ray
initialisation).
"""
from GridKIT.core.config import settings
from GridKIT.rl_engine.convergence import ConvergenceTracker
from GridKIT.rl_engine.trainer import Trainer


def run_trace(tracker: ConvergenceTracker, rewards: list[float]):
    """Feed rewards one iteration at a time; stop at the first StopDecision."""
    decisions = []
    for i, r in enumerate(rewards, start=1):
        d = tracker.should_stop(i, r)
        decisions.append(d)
        if d.stop:
            break
    return decisions


# ── ConvergenceTracker ─────────────────────────────────────────

def test_plateau_stops_after_patience():
    t = ConvergenceTracker(patience=3, min_delta=1e-3, min_iterations=0, max_iterations=20, smooth_window=1)
    # climbs to 10.0 by iteration 3, then flat → stop at 3 + patience = 6
    decisions = run_trace(t, [1.0, 5.0, 10.0, 10.0, 10.0, 10.0])
    assert decisions[-1].stop
    assert decisions[-1].reason == "converged"
    assert len(decisions) == 6


def test_improvement_resets_patience():
    t = ConvergenceTracker(patience=3, min_delta=1e-3, min_iterations=0, max_iterations=20, smooth_window=1)
    # a real improvement at iteration 4 (10.2 > 10.0) restarts the plateau clock
    decisions = run_trace(t, [10.0, 10.0, 10.0, 10.2, 10.2, 10.2, 10.2])
    assert len(decisions) == 7          # 4 (reset) + patience 3
    assert decisions[-1].reason == "converged"


def test_small_improvements_do_not_reset_patience():
    t = ConvergenceTracker(patience=2, min_delta=1e-3, min_iterations=0, max_iterations=20, smooth_window=1)
    # 0.0005 jumps stay below min_delta → plateau counts from iteration 1
    decisions = run_trace(t, [10.0, 10.0005, 10.0004, 10.0003])
    assert decisions[-1].stop
    assert len(decisions) == 3          # 1 (best) + patience 2


def test_no_stop_before_min_iterations():
    t = ConvergenceTracker(patience=2, min_delta=1e-3, min_iterations=50, max_iterations=100, smooth_window=1)
    decisions = run_trace(t, [10.0] * 10)
    assert not any(d.stop for d in decisions)
    assert len(decisions) == 10


def test_convergence_never_precedes_min_iterations():
    t = ConvergenceTracker(patience=2, min_delta=1e-3, min_iterations=5, max_iterations=20, smooth_window=1)
    decisions = run_trace(t, [10.0] * 12)
    assert not decisions[3].stop        # iteration 4 < min 5
    assert decisions[4].stop            # iteration 5 → converged
    assert decisions[4].reason == "converged"


def test_max_iterations_stops_without_convergence():
    t = ConvergenceTracker(patience=5, min_delta=1e-3, min_iterations=0, max_iterations=4, smooth_window=1)
    decisions = run_trace(t, [-5.0, -4.9, -4.8, -4.7])   # still improving
    assert decisions[-1].stop
    assert decisions[-1].reason == "max_iterations"


def test_monotone_improvement_never_stops_below_cap():
    t = ConvergenceTracker(patience=2, min_delta=1e-3, min_iterations=0, max_iterations=10, smooth_window=1)
    decisions = run_trace(t, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    assert len(decisions) == 6
    assert not any(d.stop for d in decisions)


# ── Agent-count batch scaling (pure helpers, no Ray) ──────────

def _trainer():
    return Trainer(env_factory=None, config_func=lambda env_name: object())


def test_reference_is_minimal_stub_agent_count():
    # data/stub_network_minimal.json = 2 households × 3 devices (EV+battery+hp)
    assert settings.ippo_batch_scaling_ref_agents == 6


def test_scaling_factor_floor_and_ceiling():
    t = _trainer()
    assert t._scaling_factor(0) == 1     # unknown count → unscaled
    assert t._scaling_factor(1) == 1     # fewer agents than reference
    assert t._scaling_factor(6) == 1     # reference itself → untuned defaults
    assert t._scaling_factor(7) == 2     # one over the reference → doubles
    assert t._scaling_factor(13) == 3


def test_scaled_batch_sizes_only_grow():
    t = _trainer()
    assert t._scaled_batch_sizes(0) == (None, None)     # no scaling
    assert t._scaled_batch_sizes(6) == (None, None)     # factor 1 → pass through
    train_batch, minibatch = t._scaled_batch_sizes(12)  # factor 2
    assert train_batch == settings.ippo_train_batch_size * 2
    assert minibatch == settings.ippo_minibatch_size * 2


# ── Real pipeline trace (64-agent run, reward ~ constant) ──────
# Data exported from a real 64-agent pipeline run: episode_return_mean per
# iteration. The reward is essentially flat but NOISY (amplitude ≈ ±20 around
# ≈ −370, min −392.85 / max −351.09). These tests pin down *when* the early
# stopper fires on this exact trace and why — so the "only 13 of 20
# iterations" behaviour is reproducible and explainable, not mysterious.

REAL_64_AGENT_RUN_MEANS = [
    -362.7965751925,
    -369.4279833072,
    -373.8791510989,
    -360.9127992123,
    -381.1899859962,
    -384.3414059046,
    -351.0938403148,  # best mean → iteration 7
    -367.2642383444,
    -369.8785702039,
    -378.1098743252,
    -356.9709686938,
    -392.8512877015,
    -370.0233490333,
]


def test_real_pipeline_trace_stops_at_iteration_13():
    # Pipeline defaults: patience 6, min_delta 1e-3, min_iterations 5, cap 20.
    # Best mean = iteration 7 → plateau fires at 7 + 6 = iteration 13.
    tracker = ConvergenceTracker(
        patience=6, min_delta=1e-3, min_iterations=5, max_iterations=20,
        smooth_window=1,
    )
    decisions = run_trace(tracker, REAL_64_AGENT_RUN_MEANS)

    assert len(decisions) == 13
    assert decisions[-1].stop
    assert decisions[-1].reason == "converged"


def test_real_pipeline_trace_noisier_min_delta_stops_sooner():
    # With min_delta=15 (≈ noise amplitude) no reward ever beats the first best
    # by that margin → patience counts straight from iteration 1's best and the
    # stopper fires EARLIER (iteration 7), not later. That is the counter
    # intuitive consequence of the tiny default min_delta: nearly every wiggle
    # resets the clock, so the stop point is driven by RANDOM noise, not by a
    # genuine plateau.
    tracker = ConvergenceTracker(
        patience=6, min_delta=15.0, min_iterations=5, max_iterations=20,
        smooth_window=1,
    )
    decisions = run_trace(tracker, REAL_64_AGENT_RUN_MEANS)

    assert len(decisions) == 7
    assert decisions[-1].stop
    assert decisions[-1].reason == "converged"


def test_real_pipeline_trace_larger_patience_extends_the_run():
    # patience 8 (same data, same noise): the stopper may only fire at the
    # earliest at best(7) + 8 = iteration 15. Fill the trace beyond the 13 real
    # points with a flat continuation to see the plateau criterion actually fire.
    trace = REAL_64_AGENT_RUN_MEANS + [-370.0] * 10
    tracker = ConvergenceTracker(
        patience=8, min_delta=1e-3, min_iterations=5, max_iterations=20,
        smooth_window=1,
    )
    decisions = run_trace(tracker, trace)

    assert len(decisions) == 15
    assert decisions[-1].stop
    assert decisions[-1].reason == "converged"


def test_smoothed_window_is_sliding_average_not_cumulative():
    # k=3-Fenster über [1..6]: der geglättete Wert ist der Durchschnitt der
    # letzten k per-Iteration-Mittelwerte. Das ist weder das Mittel ALLER Werte
    # seit Trainingsbeginn (it6 wäre dann 3.5 statt 5.0) noch ein nochmaliges
    # Mitteln bereits gemittelter Fensterwerte (it6 = (4+5+6)/3 = 5.0). Bis das
    # Fenster gefüllt ist, wird mit den verfügbaren Werten gemittelt.
    tracker = ConvergenceTracker(smooth_window=3)
    expected = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    for i, r in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], start=1):
        tracker.should_stop(i, r)
        assert tracker.smoothed_reward == expected[i - 1]


def test_real_pipeline_trace_smoothing_removes_noise_peak_influence():
    # Derselbe 64-Agent-Lauf, gleiche Parameter — nur das Fenster ändert sich:
    # window=1 (Rohwerte) → der Rauschpeak in Iteration 7 (−351.09) setzt das
    # polit-beste Niveau auf Iteration 7 → Stop bei 7 + 6 = 13. window=4 → die
    # geglättete Kurve hebt nie über das Niveau von Iteration 1 (−362.8) → Stop
    # deterministisch bei 1 + patience = 7. Der Stopppunkt hängt also nicht mehr
    # davon ab, wo das Reward-Rauschen zufällig seinen höchsten Punkt hat.
    def run(window):
        tracker = ConvergenceTracker(
            patience=6, min_delta=1e-3, min_iterations=5, max_iterations=20,
            smooth_window=window,
        )
        return run_trace(tracker, REAL_64_AGENT_RUN_MEANS)

    unsmoothed = run(1)
    smoothed = run(4)

    assert len(unsmoothed) == 13          # 7 (Rauschpeak) + patience 6
    assert unsmoothed[-1].reason == "converged"
    assert len(smoothed) == 7             # 1 (geglättetes best) + patience 6
    assert smoothed[-1].reason == "converged"
    assert smoothed[-1].stop


def test_smoothing_keeps_late_single_lucky_peak_from_resetting_the_clock():
    # Der Verlauf erreicht früh ein echtes (geglättetes) Plateau bei −10
    # (Iteration 3) und rauscht danach wieder runter — mit einem EINZELNEN
    # Glücks-Peak von roh +5 (Iteration 5). Ohne Smoothing (window=1) setzt der
    # rohe Peak das best auf +5 → die Plateau-Uhr springt auf Iteration 5, der
    # Stopp verzögert sich auf 5 + patience. Mit window=2 ist der Peak gedämpft
    # (avg(−90, +5) = −42.5, unter dem Plateau-Niveau −10) → kein Reset.
    trace = [-20.0, -10.0, -10.0, -90.0, 5.0, -90.0, -90.0, -90.0, -90.0, -90.0]

    def run(window):
        tracker = ConvergenceTracker(
            patience=4, min_delta=1e-3, min_iterations=0, max_iterations=30,
            smooth_window=window,
        )
        return run_trace(tracker, trace)

    unsmoothed = run(1)
    smoothed = run(2)

    # Roh: best bei it5 (+5) → 5 + 4 = 9 (der Glücks-Peak verzögert den Stop).
    assert len(unsmoothed) == 9
    assert unsmoothed[-1].reason == "converged"
    # Geglättet: best bei it3 (−10) → 3 + 4 = 7; der Einzelpeak zählt nicht.
    assert len(smoothed) == 7
    assert smoothed[-1].reason == "converged"
    assert smoothed[-1].stop


def test_real_pipeline_trace_flat_signal_stops_at_iteration_7():
    # A PERFECTLY flat trace (a textbook plateau) picks its best reward at
    # iteration 1 → the stopper fires at max(min, 1 + patience) = list at 7.
    # The real 64-agent trace (noise around a flat mean) fires later at 13,
    # because the noise briefly "improved" the best to iteration 7. Same
    # plateau — different stop point purely due to random noise. This test
    # documents that the tracker cannot distinguish "converged" (flat) from
    # "still noisy" (flat + noise), so the stop point is partly luck.
    flat = [-370.0] * 13
    tracker = ConvergenceTracker(
        patience=6, min_delta=1e-3, min_iterations=5, max_iterations=20,
        smooth_window=1,
    )
    decisions = run_trace(tracker, flat)

    assert len(decisions) == 7
    assert decisions[-1].reason == "converged"
