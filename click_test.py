#!/usr/bin/env python3
import click

def mutate_other(ctx, param, value):
    """BAD: this callback writes into another option."""
    if value is not None:
        ctx.params["param1"] = f"(from param2: {value})"
    return None   # because expose_value=False

@click.group()
def cli():
    pass

@cli.command()
@click.option("--param1", default="DEFAULT1")
@click.option("--param2", callback=mutate_other, expose_value=False)
def demo(param1):
    """Print what param1 ends up as."""
    click.echo(f"param1 seen by function = {param1!r}")

if __name__ == "__main__":
    cli()
