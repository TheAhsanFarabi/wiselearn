"""
Reporter — the 'teaching voice' of wiselearn.

All printed output goes through this module. This ensures a consistent
tone across the library and makes it easy to add a `quiet=True` mode later.

Uses 'rich' for colored output if available; falls back to plain print otherwise.
"""
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class _PlainConsole:
    """Drop-in stand-in for rich.Console when rich isn't installed."""

    @staticmethod
    def _strip_markup(text: str) -> str:
        import re
        return re.sub(r"\[/?[a-zA-Z0-9 _]+\]", "", text)

    def print(self, msg: str = "") -> None:
        print(self._strip_markup(msg))


class Reporter:
    """Handles all printed output. Single source of truth for the library's voice."""

    def __init__(self, quiet: bool = False):
        if _HAS_RICH:
            # Use terminal-style output even in Jupyter (no HTML bloat).
            # force_terminal=True keeps colors; force_jupyter=False prevents
            # the oversized HTML rendering.
            self.console = Console(
                force_terminal=True,
                force_jupyter=False,
                color_system="standard",
            )
        else:
            self.console = _PlainConsole()
        self.quiet = quiet

    def section(self, emoji: str, title: str) -> None:
        """Start a new section, e.g. '📊 Data inspection'."""
        if self.quiet:
            return
        self.console.print()
        self.console.print(f"[bold cyan]{emoji} {title}[/bold cyan]")

    def info(self, msg: str) -> None:
        """Print a regular info line."""
        if self.quiet:
            return
        self.console.print(f"   • {msg}")

    def detail(self, msg: str) -> None:
        """Print a sub-detail (indented further)."""
        if self.quiet:
            return
        self.console.print(f"     [dim]{msg}[/dim]")

    def warn(self, msg: str, fix: str | None = None) -> None:
        """Print a warning. Optionally include a suggested fix."""
        if self.quiet:
            return
        self.console.print(f"   [yellow]⚠️  {msg}[/yellow]")
        if fix:
            self.console.print(f"     [dim yellow]→ {fix}[/dim yellow]")

    def critical(self, msg: str, fix: str | None = None) -> None:
        """Print a critical issue. Loud, hard to miss."""
        if self.quiet:
            return
        self.console.print()
        if _HAS_RICH:
            self.console.print(Panel(
                Text.from_markup(
                    f"[bold red]🚨 {msg}[/bold red]"
                    + (f"\n\n[red]{fix}[/red]" if fix else "")
                ),
                border_style="red",
                title="[bold red]CRITICAL[/bold red]",
            ))
        else:
            self.console.print("=" * 60)
            self.console.print(f"🚨 CRITICAL: {msg}")
            if fix:
                self.console.print("")
                self.console.print(fix)
            self.console.print("=" * 60)

    def explain(self, msg: str) -> None:
        """Print a teaching moment — explains WHY something is happening."""
        if self.quiet:
            return
        self.console.print(f"   [magenta]💡 {msg}[/magenta]")

    def success(self, msg: str) -> None:
        """Print a success message."""
        if self.quiet:
            return
        self.console.print(f"   [green]✅ {msg}[/green]")

    def next_step(self, msg: str) -> None:
        """Suggest the user's next action."""
        if self.quiet:
            return
        self.console.print(f"   [dim]→ Next: {msg}[/dim]")

    def blank(self) -> None:
        """Just a blank line."""
        if self.quiet:
            return
        self.console.print()


# Module-level singleton — used across the library
reporter = Reporter()