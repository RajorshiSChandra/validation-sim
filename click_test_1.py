#!/usr/bin/env python3
import click

def id_callback(ctx, param, value):
    # Just return the value (or transform/validate it)
    return value

@click.group()
def cli():
    pass

@cli.command()
@click.option("--param1", default=None)
@click.option("--param2", callback=id_callback, expose_value=True, default=None)
def demo(param1, param2):
    click.echo(f"raw param1={param1!r}, raw param2={param2!r}")
    # precedence rule: param2 overrides param1 if present
    if param2 is not None:
        param1 = f"(from param2: {param2})"
    if param1 is None:
        raise click.BadParameter("Need --param1 or --param2")
    click.echo(f"effective param1 = {param1!r}")

if __name__ == "__main__":
    cli()
