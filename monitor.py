from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from pathlib import Path
import typer
from core import utils

cns = Console()
cprint = cns.print


def main(
    sky_model: list[str] | None = None,
    nchunks: list[int] | None = None,
    layout: list[str] | None = None,
    prefix: list[str] | None = None,
    show_redundant: bool = True,
    show_nonred: bool = True,
    channels: list[int] | None = None,
    chunks: list[int] | None = None,
    hide_not_run: bool = False,
    quiet: bool = False,
    max_prints: int = 10
):
    logdir = utils.LOGDIR / 'vis'
    outdir = utils.OUTDIR

    def globify_options(options, format='', allow_omission=False):
        if options:
            if len(options) == 1:
                return f"{options[0]:{format}}"
            else:
                return "(" + "|".join(f"{o:{format}}" for o in options) + ")"
        else:
            return '**' if allow_omission else "*"

    if channels is None:
        channels = list(range(1536))

    sky_glob = globify_options(sky_model)
    nchunks_glob = globify_options(nchunks, format="05d")
    layout_glob = globify_options(layout)
    prefix_glob = globify_options(prefix, allow_omission=True)
    if show_redundant and show_nonred:
        redglob="*"
    elif show_redundant:
        redglob="red"
    elif show_nonred:
        redglob="nonred"
    else:
        raise ValueError("You can't not show redundant and not redundant.")

    fmt = utils.DIRFMT.replace("{chunks:05d}", "{chunks}")  # to use string glob
    modelglob = fmt.format(
        sky_model=sky_glob, prefix=prefix_glob, chunks=nchunks_glob, layout=layout_glob,
        redundant=redglob
    )
    
    cprint(f"Model glob: {modelglob}")


    all_log_models = [pth.relative_to(logdir) for pth in sorted(logdir.glob(modelglob))]
    all_out_models = [pth.relative_to(outdir) for pth in sorted(outdir.glob(modelglob))]
    
    for mdl in all_log_models:
        parameters = utils.parse_direc(mdl)

        cprint(Rule(str(mdl)))

        # Exit early from this model if no outputs exist.
        if mdl not in all_out_models:
            cprint("[red]No outputs found...")
            break

        chunks_list = list(range(parameters['chunks'])) if chunks is None else chunks
        files = []
        for channel in channels:
            for chunk in chunks_list:
                fname = utils.get_file(chunk=chunk, channel=channel, with_dir=False).name
                # Get the *latest* logfile
                logfl = sorted((logdir / mdl).glob(f"{fname}-*.out"))
                logfl = logfl[-1] if len(logfl)>0 else None

                # Get the only output
                outfl = sorted((outdir / mdl).glob(f"{fname}.uvh5"))
                outfl = outfl[0] if outfl else None
                files.append((channel, chunk, logfl, outfl))
    
        npassed = len([x for x in files if x[2] is not None and x[3] is not None])
        cprint(f"[green]{npassed} files are completed properly.")

        if run_without_log := [
            x[3] for x in files if x[2] is None and x[3] is not None
        ]:
            cprint(f"[orange]{len(run_without_log)} files are complete, but have no log:")
            for x in run_without_log:
                cprint(f"\t{x.name}")
            cprint()


        if run_with_error := [
            x[2] for x in files if x[2] is not None and x[3] is None
        ]:
            cprint(f"[red]{len(run_with_error)} files errored:")
            run_with_error = run_with_error[:max_prints]
            
            for x in run_with_error:
                cprint(f"\t{x.relative_to(Path(__file__).parent)}")
            cprint()
    
            with open(run_with_error[-1], 'r') as fl:
                last_ten_lines = fl.readlines()[-10:]            
                cprint(Panel("\n".join(last_ten_lines), title="Error Message"))
            

        if not hide_not_run:
            not_yet_run = [x[:2] for x in files if x[2] is None and x[3] is None]
            cprint(f"{len(not_yet_run)} files not yet run at all:")
            if not quiet and not_yet_run:
                for x in not_yet_run:
                    cprint(f"channel = {x[0]:04d}, chunk = {x[1]:04d}")
        cprint()
            
if __name__ == '__main__':
    typer.run(main)
