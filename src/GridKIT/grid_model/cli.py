# grid_model/cli.py
"""
Command-line tool for building/inspecting/editing a GridNetwork step by step,
persisting it as JSON between invocations (GridNetwork.model_dump_json /
model_validate_json — no separate serialization logic needed).

Examples:
    # Real GridCreator network (OSM step, via conda), cut down to one transformer's feeder
    python -m grid_model.cli build --top 52.396954 --bottom 52.390330 \\
        --left 13.267724 --right 13.273824 --scenario "South Berlin" \\
        --single-feeder --max-households 15 --out network.json

    # Real GridCreator network from the ding0 archive (step 1 only, no conda env
    # needed — real, individually-sized transformers instead of OSM's one generic one)
    python -m grid_model.cli build --ding0 \\
        --south 49.098 --west 8.703 --north 49.112 --east 8.720 --out network.json

    # Or the stub network, for a fast demo without GridCreator
    python -m grid_model.cli build --stub --out network.json

    python -m grid_model.cli plot network.json --out-dir outputs/cli_demo

    python -m grid_model.cli add-bus network.json --bus-id new_bus \\
        --connect-to bus_4 --x 6.0 --y 0.0
    python -m grid_model.cli add-household network.json --bus-id new_bus

    python -m grid_model.cli plot network.json --out-dir outputs/cli_demo2

`add-bus`/`add-household`/`add-ev` overwrite the input file by default
(--out to write elsewhere instead), so commands chain naturally.
"""

import argparse
from pathlib import Path

from core.models import BusModel, GridNetwork, LineModel
from grid_model.builder import FixedNetworkBuilder, OSMNetworkBuilder, StubNetworkBuilder
from grid_model.environment import GridEnv
from grid_model.episode_plots import run_episode, save_plots
from grid_model import gridcreator_loader
from grid_model import network_editor

# Defaults for a manually added line, matching data/stub_network.json's own values.
DEFAULT_LINE_R_OHM_PER_KM = 0.32
DEFAULT_LINE_X_OHM_PER_KM = 0.08
DEFAULT_LINE_MAX_I_KA = 0.045
DEFAULT_LINE_LENGTH_KM = 0.05


def _load_network(path: str) -> GridNetwork:
    return GridNetwork.model_validate_json(Path(path).read_text())


def _save_network(network: GridNetwork, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(network.model_dump_json(indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    if args.stub:
        network = StubNetworkBuilder().build()
    elif args.ding0:
        missing = [name for name in ("south", "west", "north", "east") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"build --ding0: --{'/--'.join(missing)} required")
        try:
            network = gridcreator_loader.build_grid_network_from_ding0(
                args.south, args.west, args.north, args.east,
                grids_dir=args.grids_dir, residential_only=not args.include_non_residential,
            )
        except gridcreator_loader.Ding0Unavailable as e:
            raise SystemExit(f"build --ding0: {e}")
        print(f"Full network: {len(network.buses)} buses, {network.n_households} households, "
              f"{len(network.transformers)} transformers (real, individually-sized).")
    else:
        missing = [name for name in ("top", "bottom", "left", "right", "scenario") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"build: --{'/--'.join(missing)} required unless --stub or --ding0 is set")

        builder = OSMNetworkBuilder(
            top=args.top, bottom=args.bottom, left=args.left, right=args.right,
            scenario=args.scenario, conda_env=args.conda_env,
        )
        network = builder.build()
        print(f"Full network: {len(network.buses)} buses, {network.n_households} households, "
              f"{len(network.transformers)} transformers.")
        if len(network.transformers) > 1:
            print("  households per feeder:", builder.feeder_household_counts(network))

        if args.single_feeder:
            network = builder.build_single_feeder(max_households=args.max_households, network=network)

    _save_network(network, args.out)
    print(f"Built {network.network_id!r}: {len(network.buses)} buses, {network.n_households} households, "
          f"{len(network.transformers)} transformers. Saved to {args.out}")


def cmd_plot(args: argparse.Namespace) -> None:
    network = _load_network(args.network)
    env = GridEnv(builder=FixedNetworkBuilder(network))
    log = run_episode(env, seed=args.seed)
    out_dir = Path(args.out_dir)
    save_plots(log, network, out_dir)
    print(f"Plots saved to {out_dir}")


def cmd_add_bus(args: argparse.Namespace) -> None:
    network = _load_network(args.network)
    bus = BusModel(bus_id=args.bus_id, v_nom_kv=args.v_nom_kv, x_coord=args.x, y_coord=args.y)

    line = None
    if args.connect_to:
        line = LineModel(
            line_id=args.line_id or f"{args.connect_to}_to_{args.bus_id}",
            from_bus=args.connect_to,
            to_bus=args.bus_id,
            length_km=args.length_km,
            r_ohm_per_km=args.r_ohm_per_km,
            x_ohm_per_km=args.x_ohm_per_km,
            max_i_ka=args.max_i_ka,
        )

    network = network_editor.add_bus(network, bus, line)
    out = args.out or args.network
    _save_network(network, out)
    suffix = f", connected to {args.connect_to!r}" if args.connect_to else " (unconnected)"
    print(f"Added bus {args.bus_id!r}{suffix}. Saved to {out}")


def cmd_add_household(args: argparse.Namespace) -> None:
    network = _load_network(args.network)
    network = network_editor.add_household(network, args.bus_id, seed=args.seed)
    out = args.out or args.network
    _save_network(network, out)
    print(f"Marked {args.bus_id!r} as a household. Saved to {out}")


def cmd_add_ev(args: argparse.Namespace) -> None:
    network = _load_network(args.network)
    network = network_editor.add_ev(network, args.bus_id, seed=args.seed)
    out = args.out or args.network
    _save_network(network, out)
    print(f"Added EV availability to {args.bus_id!r}. Saved to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grid_model.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="build a GridNetwork (GridCreator, ding0, or stub) and save it as JSON")
    p_build.add_argument("--out", required=True, help="path to write the network JSON to")
    p_build.add_argument("--stub", action="store_true", help="use the stub network instead of GridCreator")
    # OSM (via conda-run GridCreator)
    p_build.add_argument("--top", type=float)
    p_build.add_argument("--bottom", type=float)
    p_build.add_argument("--left", type=float)
    p_build.add_argument("--right", type=float)
    p_build.add_argument("--scenario", type=str)
    p_build.add_argument("--conda-env", type=str, default="GridCreator")
    p_build.add_argument("--single-feeder", action="store_true", help="extract a single transformer's feeder (see build_single_feeder)")
    p_build.add_argument("--max-households", type=int, default=None)
    # ding0 archive (GridCreator step 1 only — no conda env needed)
    p_build.add_argument("--ding0", action="store_true",
                         help="build from the local ding0 grid archive instead of running GridCreator/OSM — "
                              "real, individually-sized transformers (see grid_model/gridcreator_loader.py)")
    p_build.add_argument("--south", type=float, help="ding0: bbox south latitude")
    p_build.add_argument("--west", type=float, help="ding0: bbox west longitude")
    p_build.add_argument("--north", type=float, help="ding0: bbox north latitude")
    p_build.add_argument("--east", type=float, help="ding0: bbox east longitude")
    p_build.add_argument("--grids-dir", type=str, default=None,
                         help="ding0: archive location (default: $GRIDKIT_DING0_GRIDS_DIR or the vendored default)")
    p_build.add_argument("--include-non-residential", action="store_true",
                         help="ding0: keep non-residential loads as household buses too (default: residential only)")
    p_build.set_defaults(func=cmd_build)

    p_plot = subparsers.add_parser("plot", help="run one episode on a saved network and save plots")
    p_plot.add_argument("network", help="path to a network JSON file")
    p_plot.add_argument("--out-dir", default="outputs/cli_demo")
    p_plot.add_argument("--seed", type=int, default=42)
    p_plot.set_defaults(func=cmd_plot)

    p_add_bus = subparsers.add_parser("add-bus", help="add a new bus, optionally connected in via a new line")
    p_add_bus.add_argument("network", help="path to a network JSON file")
    p_add_bus.add_argument("--bus-id", required=True)
    p_add_bus.add_argument("--v-nom-kv", type=float, default=0.4)
    p_add_bus.add_argument("--x", type=float, default=None, help="x coordinate (longitude for real networks)")
    p_add_bus.add_argument("--y", type=float, default=None, help="y coordinate (latitude for real networks)")
    p_add_bus.add_argument("--connect-to", default=None, help="existing bus id to connect the new bus to")
    p_add_bus.add_argument("--line-id", default=None)
    p_add_bus.add_argument("--length-km", type=float, default=DEFAULT_LINE_LENGTH_KM)
    p_add_bus.add_argument("--r-ohm-per-km", type=float, default=DEFAULT_LINE_R_OHM_PER_KM)
    p_add_bus.add_argument("--x-ohm-per-km", type=float, default=DEFAULT_LINE_X_OHM_PER_KM)
    p_add_bus.add_argument("--max-i-ka", type=float, default=DEFAULT_LINE_MAX_I_KA)
    p_add_bus.add_argument("--out", default=None, help="defaults to overwriting the input file")
    p_add_bus.set_defaults(func=cmd_add_bus)

    p_add_household = subparsers.add_parser("add-household", help="mark an existing bus as a household with a default load profile")
    p_add_household.add_argument("network", help="path to a network JSON file")
    p_add_household.add_argument("--bus-id", required=True, help="must already exist — use add-bus first if not")
    p_add_household.add_argument("--seed", type=int, default=None)
    p_add_household.add_argument("--out", default=None, help="defaults to overwriting the input file")
    p_add_household.set_defaults(func=cmd_add_household)

    p_add_ev = subparsers.add_parser("add-ev", help="attach a synthetic EV availability profile to a household bus")
    p_add_ev.add_argument("network", help="path to a network JSON file")
    p_add_ev.add_argument("--bus-id", required=True, help="must already be a household — use add-household first")
    p_add_ev.add_argument("--seed", type=int, default=None)
    p_add_ev.add_argument("--out", default=None, help="defaults to overwriting the input file")
    p_add_ev.set_defaults(func=cmd_add_ev)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
