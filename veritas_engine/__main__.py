"""CLI entry point for Veritas Engine."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veritas_engine.core.config import get_config, reload_config
from veritas_engine.core.constants import LAYERS, MILESTONES, SUCCESS_CRITERIA, VERSION
from veritas_engine.core.logger import get_logger
from veritas_engine.core.engine import Engine

app = typer.Typer(
    name="veritas",
    help="Veritas Engine - Autonomous Evolutionary Intelligence Agent Framework",
    rich_markup_mode="rich",
)
console = Console()
logger = get_logger("veritas.cli")


@app.callback()
def main(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Veritas Engine CLI."""
    if config:
        import os
        os.environ["VERITAS_CONFIG"] = config
        reload_config()
    if verbose:
        logger.setLevel("DEBUG")


@app.command()
def start(
    layers: Optional[str] = typer.Option(None, "--layers", "-l", help="Comma-separated layer names to start"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon"),
) -> None:
    """Start Veritas Engine."""
    cfg = get_config()
    console.print(Panel.fit(
        f"[bold cyan]{cfg.name}[/bold cyan] [dim]v{cfg.version}[/dim]\n"
        f"[dim]Autonomous Evolutionary Intelligence Framework[/dim]",
        title="[bold green]STARTING[/bold green]",
        border_style="cyan",
    ))

    async def _run():
        engine = Engine()
        await engine.start()
        console.print("[green]Engine started. Press Ctrl+C to stop.[/green]")
        try:
            while engine._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await engine.stop()
            console.print("[yellow]Engine stopped.[/yellow]")

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error("Engine error: %s", e)
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def run(
    goal: str = typer.Argument(..., help="Goal to execute"),
) -> None:
    """Run a single goal inference."""
    async def _run():
        engine = Engine()
        await engine.start()
        try:
            result = await engine.run_goal(goal)
            console.print_json(data=result)
        finally:
            await engine.stop()

    asyncio.run(_run())


@app.command()
def status() -> None:
    """Show system status."""
    async def _status():
        engine = Engine()
        await engine.start()
        try:
            system_status = await engine.get_status()
            table = Table(title="Veritas Engine Status", show_header=True, header_style="bold cyan")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            for layer, state in system_status.layer_status.items():
                table.add_row(layer, state)
            console.print(table)

            if system_status.emotional_state:
                es = system_status.emotional_state
                console.print(f"\n[bold magenta]Emotional State[/bold magenta]")
                console.print(f"  Curiosity: {es.curiosity:.3f}")
                console.print(f"  Urgency: {es.urgency:.3f}")
                console.print(f"  Frustration: {es.frustration:.3f}")
                console.print(f"  Achievement: {es.achievement:.3f}")
                console.print(f"  Epsilon: {es.epsilon:.3f}")
        finally:
            await engine.stop()

    asyncio.run(_status())


@app.command()
def audit() -> None:
    """Run cognitive audit."""
    async def _audit():
        engine = Engine()
        await engine.start()
        try:
            from veritas_engine.core.models import EmotionalState
            report = await engine.oracle.conduct_audit(
                engine._strategies,
                engine.daemon.get_state(),
                engine.noosphere.working_memory.summary(),
            )
            console.print(Panel(
                f"Tasks reviewed: {report.tasks_reviewed}\n"
                f"Invalid explorations: {len(report.invalid_explorations)}\n"
                f"Biases detected: {len(report.biases_detected)}\n"
                f"Improvements: {len(report.improvements)}",
                title="[bold cyan]Cognitive Audit Report[/bold cyan]",
            ))
            if report.biases_detected:
                console.print("[bold red]Biases:[/bold red]")
                for b in report.biases_detected:
                    console.print(f"  - {b['type']}: {b['description']}")
            if report.improvements:
                console.print("[bold green]Improvements:[/bold green]")
                for imp in report.improvements:
                    console.print(f"  - {imp['type']}: {imp['description']}")
        finally:
            await engine.stop()

    asyncio.run(_audit())


@app.command()
def daemon_status() -> None:
    """Show daemon (emotional layer) status."""
    async def _daemon():
        engine = Engine()
        await engine.start()
        try:
            state = engine.daemon.get_state()
            table = Table(title="Daemon Status", show_header=False)
            table.add_column("Attribute", style="magenta")
            table.add_column("Value", style="white")
            table.add_row("Curiosity", f"{state.curiosity:.3f}")
            table.add_row("Urgency", f"{state.urgency:.3f}")
            table.add_row("Frustration", f"{state.frustration:.3f}")
            table.add_row("Achievement", f"{state.achievement:.3f}")
            table.add_row("Epsilon", f"{state.epsilon:.3f}")
            table.add_row("Exploration Bias", f"{state.exploration_bias:.3f}")
            table.add_row("Exploitation Bias", f"{state.exploitation_bias:.3f}")
            table.add_row("Value Function", state.value_function_str)
            console.print(table)

            console.print("\n[bold cyan]Value Weights:[/bold cyan]")
            for k, v in state.value_weights.items():
                console.print(f"  {k}: {v:.3f}")
        finally:
            await engine.stop()

    asyncio.run(_daemon())


@app.command()
def graph(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Cypher query"),
    export: Optional[str] = typer.Option(None, "--export", "-o", help="Export to file"),
) -> None:
    """Query or export knowledge graph."""
    async def _graph():
        engine = Engine()
        await engine.start()
        try:
            if query:
                results = engine.noosphere.query_graph(query)
                console.print_json(data=results)
            elif export:
                # TODO: implement graph export
                console.print(f"[yellow]Export to {export} not yet implemented[/yellow]")
            else:
                console.print("[dim]Use --query or --export[/dim]")
        finally:
            await engine.stop()

    asyncio.run(_graph())


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start API server."""
    import uvicorn
    console.print(f"[green]Starting API server on {host}:{port}[/green]")
    uvicorn.run("veritas_engine.api.server:app", host=host, port=port, reload=reload)


@app.command()
def benchmark() -> None:
    """Run performance benchmark."""
    console.print(Panel("[yellow]Benchmark placeholder[/yellow]", title="Benchmark"))


@app.command()
def info() -> None:
    """Show system information."""
    cfg = get_config()

    table = Table(title=f"{cfg.name} v{cfg.version}", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Version", cfg.version)
    table.add_row("Data Directory", cfg.data_dir)
    table.add_row("Log Directory", cfg.log_dir)
    table.add_row("LLM Provider", cfg.llm.provider)
    table.add_row("LLM Model", cfg.llm.model)
    table.add_row("Kuzu DB", cfg.noosphere.kuzu_db_path)
    table.add_row("LanceDB", cfg.noosphere.lancedb_path)

    console.print(table)


@app.command()
def milestones() -> None:
    """Show development milestones."""
    for ms in MILESTONES:
        color = ms["color"].replace("#", "")
        console.print(f"\n[bold #{color}]Phase {ms['phase']}: {ms['title']}[/bold #{color}] "
                      f"[dim]({ms['duration_weeks']} weeks)[/dim]")
        for task in ms["tasks"]:
            console.print(f"  [dim]•[/dim] {task}")


@app.command()
def criteria() -> None:
    """Show success criteria."""
    for i, c in enumerate(SUCCESS_CRITERIA, 1):
        console.print(f"\n[bold cyan]{i}. {c['title']}[/bold cyan]")
        console.print(f"   [dim]{c['description']}[/dim]")


if __name__ == "__main__":
    app()
