import logging

import click
import numpy as np

from . import utils

logger = logging.getLogger(__name__)


class IntRangeBuilder(click.IntRange):
    """A builder for integer ranges."""

    name = "integer range"

    def convert(self, value, param, ctx) -> list[int]:
        """Check that value is INT or LOW~HIGH format and return list(low-high)."""
        try:
            int(value)
            return [super().convert(value, param, ctx)]
        except Exception:
            pass

        if isinstance(value, str):
            try:
                low, high = value.split("~")
            except ValueError:
                self.fail(f"{value!r} is not in the form 'low~high'", param, ctx)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = value
        else:
            ctx.fail(f"parameter {param!r} is of unknown type: {value}")

        low = super().convert(low, param, ctx)
        high = super().convert(high, param, ctx)

        return list(range(low, high))


def combine_int_ranges(ctx, param, value) -> list[int]:
    """Combine multiple IntRangerBuilder outputs into one list."""
    # value should be a list of lists of ints
    return sorted(set(sum(value, start=[])))


def check_ants(ctx, param, value):
    """Check whether ants and layout are provided correctly together."""
    ants = combine_int_ranges(ctx, param, value)
    if not ants and not ctx.params["layout"]:
        raise click.BadParameter("You must provide --layout or --ants")
    if ants and ctx.params["layout"]:
        raise click.BadParameter("Do not provide both --layout and --ants")

    if ants:
        ctx.params["layout"] = list(ants)


def parse_channels(channels: list[int], freq_range: tuple[float, float]) -> list[int]:
    """Combine --channels and --freq-range inputs to get one set of channels."""
    freqs = utils.FREQS_DICT["H4C"] / 1e6
    if channels:
        freqs = freqs[channels]

    if freq_range:
        mask = np.logical_and(freqs >= freq_range[0], freqs < freq_range[1])
        channels = [c for c, m in zip(channels, mask) if m]

    return channels


class opts:
    """All the options for the top-level CLI."""

    layout = click.option(
        "--layout",
        default=None,
        type=click.Choice(list(utils.ANTS_DICT.keys())),
        help="A pre-defined HERA layout to use",
    )
    ants = click.option(
        "-a",
        "--ants",
        type=IntRangeBuilder(0, 350, max_open=True),
        callback=check_ants,
        default=None,
        multiple=True,
        help="ants to use as a subset of the full HERA 350",
        expose_value=False,
    )
    ideal_layout = click.option(
        "--ideal-layout/--not-ideal-layout",
        default=True,
        help="Whether to use an idealized layout that has perfect redundancy"
    )
    redundant = click.option(
        "--redundant/--not-redundant",
        default=False,
        help="whether to use only redundant baselines in the simulation (not just in writing file)"
    )
    channels = click.option(
        "-fch",
        "--channels",
        type=IntRangeBuilder(0, len(utils.FREQS_DICT["H4C"])),
        default=[(0, len(utils.FREQS_DICT["H4C"]))],
        show_default=True,
        multiple=True,
        help="Frequency channels to include. Specify as ints or 'low~high'",
        callback=combine_int_ranges,
    )
    freq_range = click.option(
        "--freq-range",
        type=click.FloatRange(),
        default=(0, np.inf),
        nargs=2,
        help="Frequency range to include (in MHz)",
    )

    sky_model = click.option(
        "-sm",
        "--sky-model",
        default="ptsrc",
        show_default=True,
        type=click.Choice(
            [d.name for d in utils.SKYDIR.glob("*") if d.name != "raw"],
            case_sensitive=True,
        ),
        help="Sky model to simulate",
    )
    n_time_chunks = click.option(
        "--n-time-chunks",
        default=3,
        show_default=True,
        help="Split the simulation into a number of time chunks",
    )
    do_time_chunks = click.option(
        "--do-time-chunks",
        default=[],
        type=IntRangeBuilder(min=0),
        multiple=True,
        help=(
            "Only run the simulation for these time chunks (useful for "
            "debugging/exploring)"
        ),
        callback=combine_int_ranges,
    )
    gpu = click.option(
        "--gpu/--cpu", default=False, show_default=True, help="Use gpu or cpu"
    )
    slurm_override = click.option(
        "-so",
        "--slurm-override",
        nargs=2,
        multiple=True,
        metavar="FLAG VALUE",
        help="Override slurm options in the hpc config (excluding job-name and output)",
    )
    skip_existing = click.option(
        "--skip-existing/--rerun-existing",
        default=True,
        show_default=True,
        help="Skip or rerun if the simulation output already exists",
    )
    force_remake_obsparams = click.option(
        "--force-remake-obsparams",
        is_flag=True,
        help="If set, remake all obsparams before running the simulations",
    )
    log_level = click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(["INFO", "WARNING", "ERROR", "DEBUG"], case_sensitive=True),
        show_default=True,
        help="Verbosity level, also pass the flag to hera-sim-vis.py",
    )
    spline_interp_order = click.option(
        "--spline-interp-order",
        default=3,
        type=int,
        show_default=True,
        help="Order of the spatial spline interpolation",
    )
    beam_interpolator = click.option(
        "--beam-interpolator",
        default="RectBivariateSpline",
        type=click.Choice(["RectBivariateSpline", "map_coordinates"]),
        show_default=True,
        help="The interpolation function to use for beam interpolation",
    )
    profile = click.option(
        "--profile/--no-profile", default=False, help="Run line-profiling"
    )
    profile_timer_unit = click.option(
        "--profile-timer-unit", default=1e-2, help="Timer unit (in sec) for profiling", type=float
    )
    dry_run = click.option(
        "-d", "--dry-run", is_flag=True, help="Pass the flag to hera-sim-vis.py"
    )
    prefix = click.option(
        "--prefix", default="", help='prefix to put in the directory name'
    )
    phase_center_name = click.option(
        "--phase-center-name", default="zenith", help='name for the phase center'
    )

    @classmethod
    def add_opts(cls, fnc, ignore=None):
        """Add all opts to a given function."""
        ignore = ignore or []
        for name, opt in reversed(cls.__dict__.items()):
            if name not in ignore and callable(opt):
                fnc = opt(fnc)
        return fnc


def _get_sbatch_program(gpu: bool, slurm_override=None):
    """Format an SBATCH program from parts."""
    conda_params = utils.HPC_CONFIG["conda"]
    module_params = utils.HPC_CONFIG["module"]
    slurm_params = utils.HPC_CONFIG["slurm"]["gpu" if gpu else "cpu"]

    if slurm_override is not None:  # Modify slurm options
        for k, v in slurm_override:
            slurm_params[k] = v

    shebang = "#!/bin/bash"
    sbatch = "\n".join([f"#SBATCH --{k}={v}" for k, v in slurm_params.items()])

    conda = """
source {conda_path}/bin/activate
conda activate {environment_name}
""".format_map(
        conda_params
    )

    module = "\n".join([f"module load {md}" for md in module_params])

    return "\n".join([shebang, sbatch, conda, module])
