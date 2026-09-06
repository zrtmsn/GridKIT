# grid_model/test_gridcreator_episode.py
"""
End-to-end smoke run: build a real network via GridCreator, run one episode
with a simple non-RL heuristic — charge FULL whenever the EV is plugged in
and below its target SoC, else OFF — and save load/curtailment plots.

Not a pytest suite (no `def test_...` here, only a `main()`), since it needs
a live GridCreator conda env and takes a while to run. Run it directly:

    PYTHONPATH=src/GridKIT .venv/bin/python -m grid_model.test_gridcreator_episode

See grid_model/cli.py for the same building blocks exposed as separate,
composable commands (build / plot / add-bus / add-household).
"""

from core.config import settings
from grid_model.builder import FixedNetworkBuilder, OSMNetworkBuilder
from grid_model.environment import GridEnv
from grid_model.episode_plots import run_episode, save_plots

# South Berlin — already verified to produce a real network rather than an
# empty one. It has 6 transformers though, and build_pypsa_network (the PyPSA
# validation path, still single-transformer-only) only ever models
# transformers[0] — so build_single_feeder() below picks just one
# transformer's radial feeder rather than silently mis-modelling the rest.
# RadialPowerFlow itself (the actual sim backend) handles multiple feeders fine.
BBOX = dict(top=52.396954, bottom=52.390330, left=13.267724, right=13.273824)
SCENARIO = "South Berlin"
MAX_HOUSEHOLDS = 15
SEED = 42


def main() -> None:
    print(f"Building network via GridCreator for scenario {SCENARIO!r}...")
    builder = OSMNetworkBuilder(**BBOX, scenario=SCENARIO)
    full_network = builder.build()
    print(f"Full network built: {len(full_network.buses)} buses, {full_network.n_households} households, "
          f"{len(full_network.transformers)} transformers.")
    print("  households per feeder:", builder.feeder_household_counts(full_network))

    network = builder.build_single_feeder(max_households=MAX_HOUSEHOLDS, network=full_network)
    print(f"Selected feeder {network.network_id!r}: {len(network.buses)} buses, {network.n_households} households.")

    env = GridEnv(builder=FixedNetworkBuilder(network))
    log = run_episode(env, seed=SEED)

    out_dir = settings.output_dir / "gridcreator_episode"
    save_plots(log, network, out_dir)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
