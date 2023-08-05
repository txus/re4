import click
import dotenv

from re4.agent import Agent

dotenv.load_dotenv()
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="1.0.0")
def cli():
    pass


@cli.command()
@click.option("--codebase", "-c", help="Path to the codebase.", required=True)
@click.option("--verbose", "-v", help="Show sources.", is_flag=True, default=False)
def chat(codebase: str, verbose: bool):
    Agent(codebase, verbose).chat()


@cli.command()
@click.option("-n", help="How many different files to research.", default=1)
@click.option("--codebase", "-c", help="Path to the codebase.", required=True)
@click.option("--verbose", "-v", help="Show sources.", is_flag=True, default=False)
def research(n: int, codebase: str, verbose: bool):
    Agent(codebase, verbose).research(n)
