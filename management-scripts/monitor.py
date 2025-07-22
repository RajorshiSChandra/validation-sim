from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from pathlib import Path
import typer
from core import utils
import os
import numpy as np

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
    max_prints: int = 10,
    chunked: bool = False,
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

    if chunked:
        check_chunked(outdir, modelglob)
        return
    
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
                if outfl is not None:
                    flsize = os.stat(outfl).st_size
                else:
                    flsize = None
                files.append((channel, chunk, logfl, outfl, flsize))
    
        npassed = len([x for x in files if x[2] is not None and x[3] is not None])
        cprint(f"[green]{npassed} files are completed properly.")

        if run_without_log := [
            x[3] for x in files if x[2] is None and x[3] is not None
        ]:
            cprint(f"[orange]{len(run_without_log)} files are complete, but have no log:")
            for x in run_without_log:
                cprint(f"\t{x.name}")
            cprint()

        # Get files that are run, but have a weird size.
        mean_size = np.median([x[4] for x in files if x[4] is not None])
        if weird_size := [
            (x[3], x[4]) for x in files if x[4] is not None and (x[4] < mean_size - 1000 or x[4] > mean_size + 1000)
        ]:
            cprint(f"[red]{len(weird_size)} files have odd sizes: (median {mean_size/1024**3:.3f} GB)")
            for outfl, flsize in weird_size:
                cprint(f"\t{outfl.relative_to(Path(__file__).parent)}: {flsize/1024**3:.3f} GB")
        
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
            
def check_chunked(
    outdir: Path,
    modelglob: str,
) -> None:
    from pyuvdata.uvdata import FastUVH5Meta

    all_models = [pth / "rechunk" for pth in sorted(outdir.glob(modelglob))]

    for mdl in all_models:
        cprint(Rule(str(mdl)))

        # Exit early from this model if no outputs exist.
        if not mdl.exists():
            cprint("[red]No chunked outputs found...")
            break

        chunked_files = sorted(mdl.glob("*.uvh5"))
        if not chunked_files:
            cprint("[red]No chunked outputs found...")
            break

        # Determine how many files we should have, based on the first
        meta = FastUVH5Meta(chunked_files[0])
        chunksize = meta.Ntimes
        nbls = meta.Nbls
        npols = meta.Npols
        nfiles_expected = 17280 // chunksize
        nfreqs_expected = 1536

        if len(chunked_files) < nfiles_expected:
            cprint(f"[red]Only {len(chunked_files)} finished out of {nfiles_expected}.")
            break

        # If we got here, we have all the files, but we should check each one.
        for fl in chunked_files[::500]:
            meta = FastUVH5Meta(fl)

            incorrect_shapes = []
            if meta.datagrp['visdata'].shape != (chunksize*nbls, nfreqs_expected, npols):
                incorrect_shapes.append(fl)

            got_zeros = []
            d = meta.datagrp['visdata'][0, :, 0]
            if np.any(d==0):
                got_zeros.append((fl, np.sum(d==0)))    

        if incorrect_shapes:
            cprint("[red]Some files have incorrect data shapes:")
            for fl in incorrect_shapes:
                cprint(f"{fl}")
                
        if got_zeros:
            cprint("[red]Some files have data that is zero for some frequencies:")
            for fl, nzeros in got_zeros:
                cprint(f"{fl}: {nzeros}")

        if not incorrect_shapes + got_zeros:
            cprint("[green]All files completed without error!")

        cprint()
if __name__ == '__main__':
    typer.run(main)
