#!/usr/bin/env python3
import argparse
import difflib
import inspect
import json
import os
import re
import readline
import sqlite3
import subprocess
import sys
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI, APIConnectionError, APIStatusError
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
import signal

load_dotenv()

# Rich console for beautiful output
console = Console()

# History file for readline
HISTORY_FILE = Path.home() / ".aigent_history"


@dataclass
class SessionState:
    """Track session state for status bar."""
    model: str = ""
    base_url: str = ""
    connected: bool = False
    total_tokens: int = 0
    request_count: int = 0
    last_request_tokens: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.last_request_tokens = tokens
        self.request_count += 1

    def get_uptime(self) -> str:
        delta = datetime.now() - self.start_time
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{mins}m"


# Global session state
session = SessionState()


# =============================================================================
# PROJECT MANAGEMENT
# =============================================================================

AIGENT_DIR = ".aigent"
PROJECT_CONFIG_FILE = "config.json"
PROJECT_MEMORY_FILE = "memory.md"


@dataclass
class ProjectConfig:
    """Project configuration and metadata."""
    name: str = ""
    description: str = ""
    tech_stack: List[str] = field(default_factory=list)
    important_files: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tech_stack": self.tech_stack,
            "important_files": self.important_files,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            tech_stack=data.get("tech_stack", []),
            important_files=data.get("important_files", []),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


# Global project state
current_project: Optional[ProjectConfig] = None
project_path: Optional[Path] = None


def get_aigent_dir(base_path: Path = None) -> Path:
    """Get the .aigent directory path."""
    if base_path is None:
        base_path = Path.cwd()
    return base_path / AIGENT_DIR


def detect_project(start_path: Path = None) -> Optional[Path]:
    """Detect if we're in a project by looking for .aigent directory."""
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    # Walk up the directory tree looking for .aigent
    while current != current.parent:
        aigent_dir = current / AIGENT_DIR
        if aigent_dir.is_dir():
            return current
        current = current.parent

    # Check root as well
    aigent_dir = current / AIGENT_DIR
    if aigent_dir.is_dir():
        return current

    return None


def load_project(path: Path) -> Optional[ProjectConfig]:
    """Load project configuration from .aigent directory."""
    global current_project, project_path

    aigent_dir = get_aigent_dir(path)
    config_file = aigent_dir / PROJECT_CONFIG_FILE

    if not config_file.exists():
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        current_project = ProjectConfig.from_dict(data)
        project_path = path
        return current_project
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load project config: {e}[/yellow]")
        return None


def save_project(config: ProjectConfig, path: Path) -> bool:
    """Save project configuration to .aigent directory."""
    aigent_dir = get_aigent_dir(path)

    try:
        aigent_dir.mkdir(parents=True, exist_ok=True)
        config.updated_at = datetime.now().isoformat()

        config_file = aigent_dir / PROJECT_CONFIG_FILE
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving project config: {e}[/red]")
        return False


def load_project_memory(path: Path) -> str:
    """Load project memory/context file."""
    aigent_dir = get_aigent_dir(path)
    memory_file = aigent_dir / PROJECT_MEMORY_FILE

    if not memory_file.exists():
        return ""

    try:
        return memory_file.read_text(encoding="utf-8")
    except Exception:
        return ""


def save_project_memory(content: str, path: Path) -> bool:
    """Save project memory/context file."""
    aigent_dir = get_aigent_dir(path)

    try:
        aigent_dir.mkdir(parents=True, exist_ok=True)
        memory_file = aigent_dir / PROJECT_MEMORY_FILE
        memory_file.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        console.print(f"[red]Error saving project memory: {e}[/red]")
        return False


def get_project_context() -> str:
    """Get project context for system prompt."""
    global current_project, project_path

    if current_project is None:
        return ""

    context_parts = []
    context_parts.append(f"PROJECT: {current_project.name}")

    if current_project.description:
        context_parts.append(f"DESCRIPTION: {current_project.description}")

    if current_project.tech_stack:
        context_parts.append(f"TECH STACK: {', '.join(current_project.tech_stack)}")

    if current_project.important_files:
        context_parts.append(f"IMPORTANT FILES: {', '.join(current_project.important_files)}")

    if current_project.notes:
        context_parts.append(f"NOTES: {current_project.notes}")

    # Load memory file
    if project_path:
        memory = load_project_memory(project_path)
        if memory:
            context_parts.append(f"PROJECT MEMORY:\n{memory}")

    return "\n".join(context_parts)


def display_project_info():
    """Display current project information."""
    global current_project, project_path

    if current_project is None:
        console.print(Panel(
            "[yellow]No project detected.[/yellow]\n\n"
            "Initialize a project with: [cyan]/project init[/cyan]",
            title="[bold]Project[/bold]",
            border_style="blue"
        ))
        return

    # Build project info display
    info_parts = []
    info_parts.append(f"[bold cyan]Name:[/bold cyan] {current_project.name}")

    if current_project.description:
        info_parts.append(f"[bold cyan]Description:[/bold cyan] {current_project.description}")

    if project_path:
        info_parts.append(f"[bold cyan]Path:[/bold cyan] {project_path}")

    if current_project.tech_stack:
        tech_str = ", ".join(f"[green]{t}[/green]" for t in current_project.tech_stack)
        info_parts.append(f"[bold cyan]Tech Stack:[/bold cyan] {tech_str}")

    if current_project.important_files:
        files_str = ", ".join(f"[blue]{f}[/blue]" for f in current_project.important_files)
        info_parts.append(f"[bold cyan]Important Files:[/bold cyan] {files_str}")

    if current_project.notes:
        info_parts.append(f"[bold cyan]Notes:[/bold cyan] {current_project.notes}")

    if current_project.created_at:
        info_parts.append(f"[dim]Created: {current_project.created_at}[/dim]")

    if current_project.updated_at:
        info_parts.append(f"[dim]Updated: {current_project.updated_at}[/dim]")

    console.print(Panel(
        "\n".join(info_parts),
        title="[bold]Project Info[/bold]",
        border_style="blue"
    ))


def display_project_memory():
    """Display project memory content."""
    global project_path

    if project_path is None:
        console.print("[yellow]No project detected.[/yellow]")
        return

    memory = load_project_memory(project_path)

    if not memory:
        console.print(Panel(
            "[dim]No memory file yet.[/dim]\n\n"
            "Add memory with: [cyan]/project memory add <text>[/cyan]\n"
            f"Or edit directly: [cyan]{get_aigent_dir(project_path) / PROJECT_MEMORY_FILE}[/cyan]",
            title="[bold]Project Memory[/bold]",
            border_style="blue"
        ))
        return

    console.print(Panel(
        Markdown(memory),
        title="[bold]Project Memory[/bold]",
        border_style="blue"
    ))


def init_project_interactive() -> bool:
    """Initialize a new project interactively."""
    global current_project, project_path

    cwd = Path.cwd()
    aigent_dir = get_aigent_dir(cwd)

    if aigent_dir.exists():
        console.print("[yellow]Project already initialized in this directory.[/yellow]")
        return False

    console.print("\n[bold cyan]Initialize New Project[/bold cyan]\n")

    # Get project name (default to directory name)
    default_name = cwd.name
    console.print(f"[dim]Project name[/dim] [{default_name}]: ", end="")
    name = input().strip() or default_name

    # Get description
    console.print("[dim]Description[/dim]: ", end="")
    description = input().strip()

    # Get tech stack
    console.print("[dim]Tech stack (comma-separated)[/dim]: ", end="")
    tech_input = input().strip()
    tech_stack = [t.strip() for t in tech_input.split(",") if t.strip()] if tech_input else []

    # Get important files
    console.print("[dim]Important files (comma-separated)[/dim]: ", end="")
    files_input = input().strip()
    important_files = [f.strip() for f in files_input.split(",") if f.strip()] if files_input else []

    # Create project config
    config = ProjectConfig(
        name=name,
        description=description,
        tech_stack=tech_stack,
        important_files=important_files,
        notes="",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )

    # Save project
    if save_project(config, cwd):
        current_project = config
        project_path = cwd

        # Create initial memory file
        initial_memory = f"# {name}\n\n## Overview\n{description}\n\n## Notes\n\n"
        save_project_memory(initial_memory, cwd)

        console.print(f"\n[green]✓[/green] Project '[cyan]{name}[/cyan]' initialized!")
        console.print(f"[dim]Config: {aigent_dir / PROJECT_CONFIG_FILE}[/dim]")
        console.print(f"[dim]Memory: {aigent_dir / PROJECT_MEMORY_FILE}[/dim]")
        return True

    return False


def handle_project_command(args: str) -> Optional[str]:
    """Handle /project commands."""
    global current_project, project_path

    parts = args.strip().split(maxsplit=1)
    subcommand = parts[0].lower() if parts else "info"
    subargs = parts[1] if len(parts) > 1 else ""

    if subcommand in ("info", ""):
        display_project_info()
        return "continue"

    elif subcommand == "init":
        init_project_interactive()
        return "continue"

    elif subcommand == "memory":
        if not subargs:
            display_project_memory()
        elif subargs.startswith("add "):
            # Add to memory
            text = subargs[4:].strip()
            if project_path and text:
                current_memory = load_project_memory(project_path)
                new_memory = current_memory + "\n" + text if current_memory else text
                if save_project_memory(new_memory, project_path):
                    console.print("[green]✓[/green] Added to project memory")
            else:
                console.print("[yellow]No project or empty text[/yellow]")
        elif subargs == "clear":
            if project_path:
                if save_project_memory("", project_path):
                    console.print("[green]✓[/green] Project memory cleared")
        elif subargs == "edit":
            if project_path:
                memory_file = get_aigent_dir(project_path) / PROJECT_MEMORY_FILE
                console.print(f"[dim]Edit: {memory_file}[/dim]")
        return "continue"

    elif subcommand == "set":
        # Set project properties
        if not current_project:
            console.print("[yellow]No project. Run /project init first.[/yellow]")
            return "continue"

        if subargs.startswith("name "):
            current_project.name = subargs[5:].strip()
            save_project(current_project, project_path)
            console.print(f"[green]✓[/green] Name set to: {current_project.name}")
        elif subargs.startswith("desc "):
            current_project.description = subargs[5:].strip()
            save_project(current_project, project_path)
            console.print(f"[green]✓[/green] Description updated")
        elif subargs.startswith("tech "):
            tech = [t.strip() for t in subargs[5:].split(",") if t.strip()]
            current_project.tech_stack = tech
            save_project(current_project, project_path)
            console.print(f"[green]✓[/green] Tech stack updated")
        elif subargs.startswith("files "):
            files = [f.strip() for f in subargs[6:].split(",") if f.strip()]
            current_project.important_files = files
            save_project(current_project, project_path)
            console.print(f"[green]✓[/green] Important files updated")
        elif subargs.startswith("notes "):
            current_project.notes = subargs[6:].strip()
            save_project(current_project, project_path)
            console.print(f"[green]✓[/green] Notes updated")
        else:
            console.print("[dim]Usage: /project set <name|desc|tech|files|notes> <value>[/dim]")
        return "continue"

    elif subcommand == "help":
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        table.add_row("/project", "Show project info")
        table.add_row("/project init", "Initialize new project")
        table.add_row("/project info", "Show project info")
        table.add_row("/project memory", "Show project memory")
        table.add_row("/project memory add <text>", "Add to project memory")
        table.add_row("/project memory clear", "Clear project memory")
        table.add_row("/project set name <name>", "Set project name")
        table.add_row("/project set desc <text>", "Set description")
        table.add_row("/project set tech <a,b,c>", "Set tech stack")
        table.add_row("/project set files <a,b>", "Set important files")
        table.add_row("/project set notes <text>", "Set notes")
        console.print(Panel(table, title="[bold]Project Commands[/bold]", border_style="blue"))
        return "continue"

    else:
        console.print(f"[yellow]Unknown project command: {subcommand}[/yellow]")
        console.print("[dim]Use /project help for available commands[/dim]")
        return "continue"


def display_status_bar():
    """Display the status bar at the bottom of the screen."""
    global current_project

    # Get terminal width
    width = console.width

    # Build status components
    model_part = f"[cyan]{session.model}[/cyan]"

    # Project name if available
    if current_project:
        project_part = f"[bold magenta]{current_project.name}[/bold magenta]"
    else:
        project_part = ""

    # Shorten path if too long
    cwd = str(Path.cwd())
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    if len(cwd) > 30:
        cwd = "..." + cwd[-27:]
    cwd_part = f"[blue]{cwd}[/blue]"

    # Token count
    if session.total_tokens > 0:
        if session.total_tokens >= 1000:
            token_str = f"{session.total_tokens / 1000:.1f}k"
        else:
            token_str = str(session.total_tokens)
        tokens_part = f"[dim]tokens:[/dim] [yellow]{token_str}[/yellow]"
    else:
        tokens_part = ""

    # Connection status
    if session.connected:
        status_part = "[green]●[/green]"
    else:
        status_part = "[red]●[/red]"

    # Uptime
    uptime_part = f"[dim]{session.get_uptime()}[/dim]"

    # Build the status line
    parts = []
    if project_part:
        parts.append(project_part)
    parts.extend([model_part, cwd_part])
    if tokens_part:
        parts.append(tokens_part)
    parts.append(status_part)
    parts.append(uptime_part)

    status_text = " │ ".join(parts)

    # Print separator and status
    console.print(Rule(style="dim"))
    console.print(f" {status_text}", highlight=False)


def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (approx 4 chars per token)."""
    return len(text) // 4


# =============================================================================
# ANIMATIONS & FEEDBACK
# =============================================================================

# Typing cursor frames for streaming animation
CURSOR_FRAMES = ["▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", " "]

# Thinking spinner frames (braille pattern animation)
THINKING_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Alternative spinners
DOTS_SPINNER = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
BRAIN_SPINNER = ["🧠", "💭", "💡", "✨"]

THINKING_MESSAGES = [
    "Thinking",
    "Analyzing",
    "Processing",
    "Considering",
    "Reasoning",
    "Pondering",
    "Evaluating",
    "Computing",
]

THINKING_DOTS = ["", ".", "..", "..."]


def play_sound(sound_type: str = "success"):
    """Play a terminal bell sound. Can be disabled via environment variable."""
    if os.environ.get("AIGENT_SOUNDS", "0") != "1":
        return

    if sound_type == "success":
        # Single short beep
        print("\a", end="", flush=True)
    elif sound_type == "error":
        # Double beep for errors
        print("\a", end="", flush=True)
        time.sleep(0.1)
        print("\a", end="", flush=True)
    elif sound_type == "complete":
        # Gentle notification
        print("\a", end="", flush=True)


class ThinkingDisplay:
    """Animated display for thinking/waiting state."""

    def __init__(self):
        self.frame_index = 0
        self.message_index = 0
        self.dots_index = 0
        self.start_time = time.time()
        self.last_message_change = time.time()

    def get_elapsed(self) -> str:
        """Get elapsed time string."""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.1f}s"

    def render(self) -> Text:
        """Render the thinking animation."""
        result = Text()

        # Get spinner frame
        spinner = THINKING_SPINNER[self.frame_index % len(THINKING_SPINNER)]
        self.frame_index += 1

        # Change message every 2 seconds
        if time.time() - self.last_message_change > 2.0:
            self.message_index = (self.message_index + 1) % len(THINKING_MESSAGES)
            self.last_message_change = time.time()

        # Animate dots
        dots = THINKING_DOTS[self.dots_index % len(THINKING_DOTS)]
        self.dots_index += 1

        message = THINKING_MESSAGES[self.message_index]

        # Build the display
        result.append(f" {spinner} ", style="bold cyan")
        result.append(message, style="bold yellow")
        result.append(dots, style="yellow")
        result.append(f"  ", style="")
        result.append(f"({self.get_elapsed()})", style="dim cyan")

        return result


class StreamingDisplay:
    """Animated display for streaming LLM responses."""

    def __init__(self, console: Console):
        self.console = console
        self.frame_index = 0
        self.start_time = time.time()

    def get_cursor(self) -> str:
        """Get the current cursor frame."""
        cursor = CURSOR_FRAMES[self.frame_index % len(CURSOR_FRAMES)]
        self.frame_index += 1
        return cursor

    def get_elapsed(self) -> str:
        """Get elapsed time string."""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.1f}s"

    def render(self, text: str, is_complete: bool = False) -> Text:
        """Render the streaming text with cursor animation."""
        result = Text()

        if is_complete:
            result.append(text, style="yellow")
        else:
            result.append(text, style="yellow")
            result.append(self.get_cursor(), style="bold cyan")

        return result

    def render_with_header(self, text: str, is_complete: bool = False) -> Group:
        """Render text with a header showing elapsed time."""
        header = Text()
        if not is_complete:
            header.append("● ", style="bold green")
            header.append("Generating ", style="dim")
            header.append(f"({self.get_elapsed()})", style="dim cyan")
        else:
            header.append("✓ ", style="bold green")
            header.append("Complete ", style="dim")
            header.append(f"({self.get_elapsed()})", style="dim cyan")

        content = self.render(text, is_complete)

        return Group(header, Text(), content)


class ToolExecutionDisplay:
    """Animated display for tool execution."""

    def __init__(self, console: Console, tool_name: str):
        self.console = console
        self.tool_name = tool_name
        self.start_time = time.time()

    def __enter__(self):
        self.status = self.console.status(
            f"[bold cyan]Executing {self.tool_name}...[/bold cyan]",
            spinner="dots"
        )
        self.status.__enter__()
        return self

    def __exit__(self, *args):
        self.status.__exit__(*args)
        elapsed = time.time() - self.start_time
        # Quick flash of completion
        self.console.print(
            f"  [dim]└─ {self.tool_name} completed in {elapsed:.2f}s[/dim]"
        )

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks.
You have access to a series of tools you can execute. Here are the tools you can execute:

{tool_list_repr}

When you want to use a tool, reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally. Use markdown formatting in your responses.
{project_context}
"""


def resolve_abs_path(path_str: str) -> Path:
    """Resolve a path string to an absolute path."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


# =============================================================================
# TOOLS
# =============================================================================

def read_file_tool(filename: str) -> Dict[str, Any]:
    """
    Gets the full content of a file provided by the user.
    :param filename: The name of the file to read.
    :return: The full content of the file.
    """
    full_path = resolve_abs_path(filename)
    with open(str(full_path), "r") as f:
        content = f.read()
    return {
        "file_path": str(full_path),
        "content": content
    }


def list_files_tool(path: str = ".") -> Dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    :param path: The path to a directory to list files from.
    :return: A list of files in the directory.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append({
            "filename": item.name,
            "type": "file" if item.is_file() else "dir"
        })
    return {
        "path": str(full_path),
        "files": all_files
    }


def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file. If old_str is empty,
    create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path to the file and the action taken.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(new_str, encoding="utf-8")
        return {
            "path": str(full_path),
            "action": "created_file"
        }
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {
            "path": str(full_path),
            "action": "old_str not found"
        }
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {
        "path": str(full_path),
        "action": "edited"
    }


def run_command_tool(command: str, working_dir: str = "") -> Dict[str, Any]:
    """
    Executes a shell command and returns the output.
    :param command: The shell command to execute.
    :param working_dir: Optional working directory for the command.
    :return: Dictionary with stdout, stderr, and return code.
    """
    cwd = resolve_abs_path(working_dir) if working_dir else Path.cwd()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120
        )
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "error": "Command timed out after 120 seconds"
        }
    except Exception as e:
        return {
            "command": command,
            "error": str(e)
        }


def search_files_tool(pattern: str, path: str = ".", file_pattern: str = "*") -> Dict[str, Any]:
    """
    Searches for a text pattern in files using grep-like functionality.
    :param pattern: The regex pattern to search for.
    :param path: The directory to search in.
    :param file_pattern: Glob pattern to filter files (e.g., "*.py").
    :return: Dictionary with matches found.
    """
    search_path = resolve_abs_path(path)
    matches = []

    for file_path in search_path.rglob(file_pattern):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for line_num, line in enumerate(content.splitlines(), 1):
                    if re.search(pattern, line):
                        matches.append({
                            "file": str(file_path),
                            "line": line_num,
                            "content": line.strip()
                        })
            except Exception:
                continue

    return {
        "pattern": pattern,
        "path": str(search_path),
        "matches": matches[:50]
    }


def find_files_tool(pattern: str, path: str = ".") -> Dict[str, Any]:
    """
    Finds files matching a glob pattern.
    :param pattern: Glob pattern to match (e.g., "*.py", "**/*.json").
    :param path: The directory to search in.
    :return: List of matching file paths.
    """
    search_path = resolve_abs_path(path)
    matches = []

    for file_path in search_path.rglob(pattern):
        matches.append({
            "path": str(file_path),
            "type": "file" if file_path.is_file() else "dir"
        })

    return {
        "pattern": pattern,
        "search_path": str(search_path),
        "matches": matches[:100]
    }


def get_cwd_tool() -> Dict[str, Any]:
    """
    Returns the current working directory.
    :return: The current working directory path.
    """
    return {"cwd": str(Path.cwd())}


def create_directory_tool(path: str) -> Dict[str, Any]:
    """
    Creates a directory (including parent directories if needed).
    :param path: The path of the directory to create.
    :return: Dictionary with the created path and status.
    """
    full_path = resolve_abs_path(path)
    try:
        full_path.mkdir(parents=True, exist_ok=True)
        return {"path": str(full_path), "action": "created"}
    except Exception as e:
        return {"path": str(full_path), "error": str(e)}


def delete_file_tool(path: str) -> Dict[str, Any]:
    """
    Deletes a file or empty directory.
    :param path: The path to delete.
    :return: Dictionary with the deleted path and status.
    """
    full_path = resolve_abs_path(path)
    try:
        if full_path.is_file():
            full_path.unlink()
            return {"path": str(full_path), "action": "deleted_file"}
        elif full_path.is_dir():
            full_path.rmdir()
            return {"path": str(full_path), "action": "deleted_directory"}
        else:
            return {"path": str(full_path), "error": "path does not exist"}
    except Exception as e:
        return {"path": str(full_path), "error": str(e)}


# =============================================================================
# GIT TOOLS
# =============================================================================

def _run_git_command(args: List[str], working_dir: str = "") -> Dict[str, Any]:
    """Helper function to run git commands."""
    cwd = resolve_abs_path(working_dir) if working_dir else Path.cwd()
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60
        )
        return {
            "command": f"git {' '.join(args)}",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Git command timed out"}
    except Exception as e:
        return {"error": str(e)}


def git_status_tool(path: str = ".") -> Dict[str, Any]:
    """
    Shows the working tree status (modified, staged, untracked files).
    :param path: The path to the git repository.
    :return: Git status output.
    """
    return _run_git_command(["status", "--short"], path)


def git_diff_tool(path: str = ".", staged: bool = False, file: str = "") -> Dict[str, Any]:
    """
    Shows changes between commits, commit and working tree, etc.
    :param path: The path to the git repository.
    :param staged: If True, show staged changes (--cached).
    :param file: Optional specific file to diff.
    :return: Git diff output.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if file:
        args.append("--")
        args.append(file)
    return _run_git_command(args, path)


def git_log_tool(path: str = ".", count: int = 10, oneline: bool = True) -> Dict[str, Any]:
    """
    Shows the commit logs.
    :param path: The path to the git repository.
    :param count: Number of commits to show.
    :param oneline: If True, show each commit on one line.
    :return: Git log output.
    """
    args = ["log", f"-{count}"]
    if oneline:
        args.append("--oneline")
    return _run_git_command(args, path)


def git_add_tool(files: str, path: str = ".") -> Dict[str, Any]:
    """
    Adds file contents to the staging area.
    :param files: Files to add (use "." for all files, or space-separated list).
    :param path: The path to the git repository.
    :return: Git add result.
    """
    args = ["add"] + files.split()
    return _run_git_command(args, path)


def git_commit_tool(message: str, path: str = ".") -> Dict[str, Any]:
    """
    Records changes to the repository.
    :param message: The commit message.
    :param path: The path to the git repository.
    :return: Git commit result.
    """
    return _run_git_command(["commit", "-m", message], path)


def git_branch_tool(path: str = ".", list_all: bool = False) -> Dict[str, Any]:
    """
    Lists, creates, or deletes branches.
    :param path: The path to the git repository.
    :param list_all: If True, list both local and remote branches.
    :return: Git branch output.
    """
    args = ["branch"]
    if list_all:
        args.append("-a")
    return _run_git_command(args, path)


def git_checkout_tool(target: str, path: str = ".", create: bool = False) -> Dict[str, Any]:
    """
    Switches branches or restores working tree files.
    :param target: Branch name or commit to checkout.
    :param path: The path to the git repository.
    :param create: If True, create a new branch (-b flag).
    :return: Git checkout result.
    """
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(target)
    return _run_git_command(args, path)


def git_pull_tool(path: str = ".", remote: str = "origin", branch: str = "") -> Dict[str, Any]:
    """
    Fetches from and integrates with another repository or local branch.
    :param path: The path to the git repository.
    :param remote: The remote to pull from.
    :param branch: The branch to pull (empty for current branch).
    :return: Git pull result.
    """
    args = ["pull", remote]
    if branch:
        args.append(branch)
    return _run_git_command(args, path)


def git_push_tool(path: str = ".", remote: str = "origin", branch: str = "") -> Dict[str, Any]:
    """
    Updates remote refs along with associated objects.
    :param path: The path to the git repository.
    :param remote: The remote to push to.
    :param branch: The branch to push (empty for current branch).
    :return: Git push result.
    """
    args = ["push", remote]
    if branch:
        args.append(branch)
    return _run_git_command(args, path)


def git_stash_tool(action: str = "list", path: str = "", message: str = "") -> Dict[str, Any]:
    """
    Stashes changes in a dirty working directory.
    :param action: Action to perform: list, push, pop, apply, drop.
    :param path: The path to the git repository.
    :param message: Optional message for stash push.
    :return: Git stash result.
    """
    args = ["stash", action]
    if action == "push" and message:
        args.extend(["-m", message])
    return _run_git_command(args, path)


def git_reset_tool(path: str = ".", mode: str = "mixed", target: str = "HEAD") -> Dict[str, Any]:
    """
    Resets current HEAD to the specified state.
    :param path: The path to the git repository.
    :param mode: Reset mode: soft, mixed, or hard.
    :param target: The commit to reset to.
    :return: Git reset result.
    """
    return _run_git_command(["reset", f"--{mode}", target], path)


# =============================================================================
# SQLITE TOOLS
# =============================================================================

def sqlite_query(database: str, query: str, params: List[Any] = None) -> Dict[str, Any]:
    """
    Executes a SELECT query on a SQLite database and returns the results.
    :param database: Path to the SQLite database file.
    :param query: The SELECT SQL query to execute.
    :param params: Optional list of parameters for parameterized queries.
    :return: Dictionary with columns and rows.
    """
    db_path = resolve_abs_path(database)
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        # Get column names
        columns = [description[0] for description in cursor.description] if cursor.description else []

        # Fetch results (limit to 1000 rows)
        rows = cursor.fetchmany(1000)
        results = [dict(row) for row in rows]

        row_count = len(results)
        has_more = cursor.fetchone() is not None

        conn.close()

        return {
            "database": str(db_path),
            "query": query,
            "columns": columns,
            "rows": results,
            "row_count": row_count,
            "truncated": has_more
        }
    except sqlite3.Error as e:
        return {"error": f"SQLite error: {e}", "query": query}
    except Exception as e:
        return {"error": str(e), "query": query}


def sqlite_execute(database: str, statement: str, params: List[Any] = None) -> Dict[str, Any]:
    """
    Executes an INSERT, UPDATE, DELETE, or CREATE statement on a SQLite database.
    :param database: Path to the SQLite database file.
    :param statement: The SQL statement to execute.
    :param params: Optional list of parameters for parameterized queries.
    :return: Dictionary with affected rows and last row id.
    """
    db_path = resolve_abs_path(database)

    try:
        # Create parent directory if needed for new databases
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        if params:
            cursor.execute(statement, params)
        else:
            cursor.execute(statement)

        conn.commit()

        result = {
            "database": str(db_path),
            "statement": statement,
            "rows_affected": cursor.rowcount,
            "last_row_id": cursor.lastrowid
        }

        conn.close()
        return result
    except sqlite3.Error as e:
        return {"error": f"SQLite error: {e}", "statement": statement}
    except Exception as e:
        return {"error": str(e), "statement": statement}


def sqlite_tables(database: str) -> Dict[str, Any]:
    """
    Lists all tables in a SQLite database.
    :param database: Path to the SQLite database file.
    :return: Dictionary with list of table names.
    """
    db_path = resolve_abs_path(database)
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "database": str(db_path),
            "tables": tables,
            "count": len(tables)
        }
    except sqlite3.Error as e:
        return {"error": f"SQLite error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def sqlite_schema(database: str, table: str = "") -> Dict[str, Any]:
    """
    Shows the schema of a SQLite database or a specific table.
    :param database: Path to the SQLite database file.
    :param table: Optional table name. If empty, shows all table schemas.
    :return: Dictionary with schema information.
    """
    db_path = resolve_abs_path(database)
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        if table:
            # Get schema for specific table
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            if not columns:
                return {"error": f"Table not found: {table}"}

            schema = {
                "table": table,
                "columns": [
                    {
                        "name": col[1],
                        "type": col[2],
                        "notnull": bool(col[3]),
                        "default": col[4],
                        "primary_key": bool(col[5])
                    }
                    for col in columns
                ]
            }

            # Get indexes
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = [{"name": idx[1], "unique": bool(idx[2])} for idx in cursor.fetchall()]
            schema["indexes"] = indexes

        else:
            # Get all table schemas
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = cursor.fetchall()

            schema = {
                "tables": [
                    {"name": t[0], "sql": t[1]}
                    for t in tables
                ]
            }

        conn.close()

        return {
            "database": str(db_path),
            "schema": schema
        }
    except sqlite3.Error as e:
        return {"error": f"SQLite error: {e}"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# WEB SEARCH TOOLS
# =============================================================================

def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo and return results.
    :param query: The search query.
    :param num_results: Number of results to return (max 10).
    :return: Dictionary with search results.
    """
    import html

    num_results = min(num_results, 10)

    try:
        # Use DuckDuckGo HTML lite version
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")

        # Parse results using regex (simple extraction)
        results = []

        # Find result blocks
        result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.+?)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.+?)</a>'

        links = re.findall(result_pattern, content)
        snippets = re.findall(snippet_pattern, content)

        for i, (link, title) in enumerate(links[:num_results]):
            result = {
                "title": html.unescape(re.sub(r'<[^>]+>', '', title)).strip(),
                "url": link,
            }
            if i < len(snippets):
                result["snippet"] = html.unescape(re.sub(r'<[^>]+>', '', snippets[i])).strip()
            results.append(result)

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except urllib.error.URLError as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_url(url: str, max_length: int = 5000) -> Dict[str, Any]:
    """
    Fetch content from a URL and return the text.
    :param url: The URL to fetch.
    :param max_length: Maximum content length to return.
    :return: Dictionary with URL content.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get("Content-Type", "")

            if "text/html" in content_type:
                html_content = response.read().decode("utf-8", errors="ignore")

                # Simple HTML to text conversion
                # Remove script and style tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                # Decode HTML entities
                import html
                text = html.unescape(text)

                return {
                    "url": url,
                    "content_type": "html",
                    "text": text[:max_length],
                    "truncated": len(text) > max_length
                }
            else:
                content = response.read().decode("utf-8", errors="ignore")
                return {
                    "url": url,
                    "content_type": content_type,
                    "text": content[:max_length],
                    "truncated": len(content) > max_length
                }

    except urllib.error.URLError as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# PYTHON PACKAGE TOOLS
# =============================================================================

def pip_list(outdated: bool = False) -> Dict[str, Any]:
    """
    List installed Python packages.
    :param outdated: If True, show only outdated packages.
    :return: Dictionary with package list.
    """
    try:
        cmd = [sys.executable, "-m", "pip", "list", "--format=json"]
        if outdated:
            cmd.append("--outdated")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {"error": result.stderr}

        packages = json.loads(result.stdout)

        return {
            "packages": packages,
            "count": len(packages),
            "outdated_only": outdated
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


def pip_show(package: str) -> Dict[str, Any]:
    """
    Show information about an installed Python package.
    :param package: The package name.
    :return: Dictionary with package information.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return {"error": f"Package not found: {package}"}

        # Parse the output
        info = {}
        for line in result.stdout.strip().split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                info[key.lower().replace("-", "_")] = value

        return {
            "package": package,
            "info": info
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


def pip_check() -> Dict[str, Any]:
    """
    Check for broken dependencies in installed packages.
    :return: Dictionary with dependency check results.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return {
                "status": "ok",
                "message": "No broken dependencies found"
            }
        else:
            # Parse broken dependencies
            issues = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    issues.append(line)

            return {
                "status": "issues_found",
                "issues": issues,
                "count": len(issues)
            }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


def pypi_info(package: str) -> Dict[str, Any]:
    """
    Get package information from PyPI (online).
    :param package: The package name.
    :return: Dictionary with PyPI package information.
    """
    try:
        url = f"https://pypi.org/pypi/{package}/json"
        req = urllib.request.Request(url)

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        info = data.get("info", {})

        return {
            "package": package,
            "version": info.get("version"),
            "summary": info.get("summary"),
            "author": info.get("author"),
            "license": info.get("license"),
            "home_page": info.get("home_page") or info.get("project_url"),
            "requires_python": info.get("requires_python"),
            "keywords": info.get("keywords"),
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"Package not found on PyPI: {package}"}
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {
    # File tools
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
    "search_files": search_files_tool,
    "find_files": find_files_tool,
    "create_directory": create_directory_tool,
    "delete_file": delete_file_tool,
    # System tools
    "run_command": run_command_tool,
    "get_cwd": get_cwd_tool,
    # Git tools
    "git_status": git_status_tool,
    "git_diff": git_diff_tool,
    "git_log": git_log_tool,
    "git_add": git_add_tool,
    "git_commit": git_commit_tool,
    "git_branch": git_branch_tool,
    "git_checkout": git_checkout_tool,
    "git_pull": git_pull_tool,
    "git_push": git_push_tool,
    "git_stash": git_stash_tool,
    "git_reset": git_reset_tool,
    # SQLite tools
    "sqlite_query": sqlite_query,
    "sqlite_execute": sqlite_execute,
    "sqlite_tables": sqlite_tables,
    "sqlite_schema": sqlite_schema,
    # Web tools
    "web_search": web_search,
    "fetch_url": fetch_url,
    # Python package tools
    "pip_list": pip_list,
    "pip_show": pip_show,
    "pip_check": pip_check,
    "pypi_info": pypi_info,
}


# =============================================================================
# TOOL FORMATTING
# =============================================================================

def get_tool_str_representation(tool_name: str) -> str:
    tool = TOOL_REGISTRY[tool_name]
    return f"""
Name: {tool_name}
Description: {tool.__doc__}
Signature: {inspect.signature(tool)}
"""


def get_full_system_prompt():
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += f"{'='*40}\nTOOL\n{'='*40}"
        tool_str_repr += get_tool_str_representation(tool_name)

    # Add project context if available
    project_ctx = get_project_context()
    if project_ctx:
        project_ctx = f"\n\n{'='*40}\nPROJECT CONTEXT\n{'='*40}\n{project_ctx}\n"
    else:
        project_ctx = ""

    return SYSTEM_PROMPT.format(tool_list_repr=tool_str_repr, project_context=project_ctx)


def parse_tool_args(tool_name: str, args_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse tool arguments from various formats.
    Handles JSON objects, single strings, positional arguments, and empty args.
    """
    args_str = args_str.strip()

    # Handle empty arguments - return empty dict for tools with default params
    if not args_str:
        return {}

    # Try JSON first
    if args_str.startswith('{'):
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            pass

    # Handle quoted string argument: 'value' or "value"
    if (args_str.startswith("'") and args_str.endswith("'")) or \
       (args_str.startswith('"') and args_str.endswith('"')):
        value = args_str[1:-1]
        # Map to first parameter of the tool
        tool = TOOL_REGISTRY.get(tool_name)
        if tool:
            sig = inspect.signature(tool)
            params = list(sig.parameters.keys())
            if params:
                return {params[0]: value}

    # Handle unquoted string (simple value)
    if args_str and not args_str.startswith('{'):
        tool = TOOL_REGISTRY.get(tool_name)
        if tool:
            sig = inspect.signature(tool)
            params = list(sig.parameters.keys())
            if params:
                return {params[0]: args_str}

    return {}


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return list of (tool_name, args) requested in various tool call formats.
    Supports:
    - tool: name({...})
    - tool: name('string')
    - [TOOL_CALLS]name({...})
    - [TOOL_CALLS]name('string')
    - <tool_call>name({...})</tool_call>
    - name({...}) where name is a known tool
    """
    invocations = []

    # Patterns to match different tool call formats
    # Capture tool name and everything inside parentheses (including empty)
    patterns = [
        # tool: name(...)
        r'tool:\s*(\w+)\s*\(([^)]*)\)',
        # [TOOL_CALLS]name(...) or [TOOL_CALL]name(...)
        r'\[TOOL_CALLS?\]\s*(\w+)\s*\(([^)]*)\)',
        # <tool_call>name(...)</tool_call>
        r'<tool_call>\s*(\w+)\s*\(([^)]*)\)\s*</tool_call>',
        # <<tool_name>>(...)
        r'<<(\w+)>>\s*\(([^)]*)\)',
    ]

    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for name, args_str in matches:
            name = name.strip()
            if name in TOOL_REGISTRY:
                args = parse_tool_args(name, args_str)
                if args is not None:
                    invocations.append((name, args))

    # Also try line-by-line for simple format: known_tool(...)
    if not invocations:
        for line in text.splitlines():
            line = line.strip()
            for tool_name in TOOL_REGISTRY:
                # Match tool_name(...) at start of line or after whitespace/bracket
                pattern = rf'(?:^|[\s\[])({tool_name})\s*\(([^)]*)\)'
                matches = re.findall(pattern, line)
                for name, args_str in matches:
                    args = parse_tool_args(name, args_str)
                    if args is not None:
                        invocations.append((name, args))

    return invocations


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_tool_call(name: str, args: Dict[str, Any]):
    """Display a tool call in a nice panel."""
    args_text = json.dumps(args, indent=2)
    content = Syntax(args_text, "json", theme="monokai", line_numbers=False)
    console.print(Panel(
        content,
        title=f"[bold cyan]Tool: {name}[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    ))


def display_tool_result(result: Dict[str, Any], name: str):
    """Display a tool result in a nice panel."""
    # Special formatting for different tool types
    if name == "list_files" and "files" in result:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        for f in result["files"]:
            icon = "" if f["type"] == "dir" else ""
            table.add_row(f"{icon} {f['filename']}", f["type"])
        console.print(Panel(table, title="[bold green]Result[/bold green]", border_style="green"))
    elif name == "read_file" and "content" in result:
        # Try to detect language from file extension
        file_path = result.get("file_path", "")
        ext = Path(file_path).suffix.lstrip(".")
        lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "json": "json", "md": "markdown"}
        lang = lang_map.get(ext, ext or "text")
        content = result["content"]
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"[bold green]{file_path}[/bold green]", border_style="green"))
    elif name == "search_files" and "matches" in result:
        if result["matches"]:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("File", style="cyan", max_width=40)
            table.add_column("Line", style="yellow", justify="right")
            table.add_column("Content", style="white")
            for m in result["matches"][:20]:
                table.add_row(
                    Path(m["file"]).name,
                    str(m["line"]),
                    m["content"][:60] + ("..." if len(m["content"]) > 60 else "")
                )
            console.print(Panel(table, title=f"[bold green]Search Results ({len(result['matches'])} matches)[/bold green]", border_style="green"))
        else:
            console.print(Panel("[yellow]No matches found[/yellow]", title="[bold green]Result[/bold green]", border_style="green"))
    elif name == "run_command" or name.startswith("git_"):
        output_parts = []
        if result.get("stdout"):
            output_parts.append(result["stdout"])
        if result.get("stderr"):
            output_parts.append(f"[red]{result['stderr']}[/red]")
        if result.get("error"):
            output_parts.append(f"[red]Error: {result['error']}[/red]")

        output = "\n".join(output_parts) if output_parts else "[dim]No output[/dim]"
        rc = result.get("return_code", "N/A")
        rc_style = "green" if rc == 0 else "red"

        # Get command for title
        cmd = result.get("command", name)
        if len(cmd) > 50:
            cmd = cmd[:47] + "..."

        # Use diff syntax highlighting for git diff
        if name == "git_diff" and result.get("stdout"):
            content = Syntax(result["stdout"][:3000], "diff", theme="monokai", line_numbers=False)
            console.print(Panel(
                content,
                title=f"[bold green]$ {cmd}[/bold green] [dim](exit: [{rc_style}]{rc}[/{rc_style}])[/dim]",
                border_style="green"
            ))
        else:
            # Show output with syntax highlighting for better visibility
            output_text = output[:3000] + ("..." if len(output) > 3000 else "")
            console.print(Panel(
                Syntax(output_text, "bash", theme="monokai", line_numbers=False) if output_parts else Text.from_markup(output),
                title=f"[bold green]$ {cmd}[/bold green] [dim](exit: [{rc_style}]{rc}[/{rc_style}])[/dim]",
                border_style="green"
            ))
    elif name.startswith("sqlite_"):
        # SQLite result display
        if result.get("error"):
            console.print(Panel(
                f"[red]{result['error']}[/red]",
                title="[bold red]SQLite Error[/bold red]",
                border_style="red"
            ))
        elif name == "sqlite_query" and "rows" in result:
            # Display query results as table
            columns = result.get("columns", [])
            rows = result.get("rows", [])

            if rows:
                table = Table(show_header=True, header_style="bold magenta", show_lines=True)
                for col in columns:
                    table.add_column(col, style="cyan")

                for row in rows[:50]:  # Limit display to 50 rows
                    table.add_row(*[str(row.get(col, "")) for col in columns])

                title = f"[bold green]Query Results[/bold green] [dim]({result.get('row_count', 0)} rows"
                if result.get("truncated"):
                    title += ", truncated"
                title += ")[/dim]"

                console.print(Panel(table, title=title, border_style="green"))
            else:
                console.print(Panel("[dim]No results[/dim]", title="[bold green]Query Results[/bold green]", border_style="green"))

        elif name == "sqlite_tables" and "tables" in result:
            # Display table list
            tables = result.get("tables", [])
            if tables:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Table Name", style="cyan")
                for t in tables:
                    table.add_row(t)
                console.print(Panel(table, title=f"[bold green]Tables ({len(tables)})[/bold green]", border_style="green"))
            else:
                console.print(Panel("[dim]No tables found[/dim]", title="[bold green]Tables[/bold green]", border_style="green"))

        elif name == "sqlite_schema" and "schema" in result:
            # Display schema info
            schema = result.get("schema", {})

            if "columns" in schema:
                # Single table schema
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Column", style="cyan")
                table.add_column("Type", style="yellow")
                table.add_column("Not Null", style="green")
                table.add_column("PK", style="red")
                table.add_column("Default", style="dim")

                for col in schema["columns"]:
                    table.add_row(
                        col["name"],
                        col["type"],
                        "✓" if col["notnull"] else "",
                        "✓" if col["primary_key"] else "",
                        str(col["default"]) if col["default"] else ""
                    )

                console.print(Panel(table, title=f"[bold green]Schema: {schema.get('table', '')}[/bold green]", border_style="green"))

            elif "tables" in schema:
                # All tables schema
                for tbl in schema["tables"]:
                    console.print(Panel(
                        Syntax(tbl["sql"], "sql", theme="monokai"),
                        title=f"[bold green]{tbl['name']}[/bold green]",
                        border_style="green"
                    ))

        elif name == "sqlite_execute":
            # Display execute result
            info = f"Rows affected: [cyan]{result.get('rows_affected', 0)}[/cyan]"
            if result.get("last_row_id"):
                info += f"\nLast row ID: [cyan]{result['last_row_id']}[/cyan]"
            console.print(Panel(info, title="[bold green]Execute Result[/bold green]", border_style="green"))

        else:
            # Fallback for other sqlite results
            result_text = json.dumps(result, indent=2)
            console.print(Panel(
                Syntax(result_text, "json", theme="monokai"),
                title="[bold green]SQLite Result[/bold green]",
                border_style="green"
            ))
    else:
        # Generic result display
        result_text = json.dumps(result, indent=2)
        if len(result_text) > 2000:
            result_text = result_text[:2000] + "\n... (truncated)"
        console.print(Panel(
            Syntax(result_text, "json", theme="monokai"),
            title="[bold green]Result[/bold green]",
            border_style="green"
        ))


def display_assistant_message(text: str):
    """Display assistant message in conversational style."""
    console.print()
    console.print("[bold cyan]aigent:[/bold cyan]")
    console.print(Markdown(text))
    console.print()


def display_welcome():
    """Display welcome message."""
    global current_project

    console.print()

    # Build welcome content
    welcome_lines = ["[bold cyan]aigent[/bold cyan] - AI Coding Assistant\n"]

    # Show project info if detected
    if current_project:
        welcome_lines.append(f"[bold magenta]Project:[/bold magenta] {current_project.name}")
        if current_project.description:
            welcome_lines.append(f"[dim]{current_project.description}[/dim]")
        welcome_lines.append("")

    welcome_lines.append("[dim]Commands:[/dim]  [cyan]/help[/cyan] [cyan]/project[/cyan] [cyan]/models[/cyan] [cyan]/tools[/cyan] [cyan]/clear[/cyan] [cyan]/exit[/cyan]")
    welcome_lines.append("[dim]Shortcuts:[/dim]  [cyan]Tab[/cyan] complete  [cyan]↑↓[/cyan] history  [cyan]Ctrl+C[/cyan] cancel  [cyan]Ctrl+D[/cyan] exit")

    console.print(Panel(
        "\n".join(welcome_lines),
        title="[bold]Welcome[/bold]",
        border_style="blue"
    ))
    console.print()


def display_help():
    """Display help information."""
    # Commands table
    cmd_table = Table(show_header=True, header_style="bold cyan", title="Commands")
    cmd_table.add_column("Command", style="cyan")
    cmd_table.add_column("Description")
    cmd_table.add_row("/help", "Show this help message")
    cmd_table.add_row("/model", "Show current model")
    cmd_table.add_row("/model <name>", "Switch to a different model")
    cmd_table.add_row("/models", "List available Ollama models")
    cmd_table.add_row("/project", "Project management (init, info, memory)")
    cmd_table.add_row("/tools", "List available tools")
    cmd_table.add_row("/clear", "Clear conversation history")
    cmd_table.add_row("/exit", "Exit the agent")

    # Shortcuts table
    shortcut_table = Table(show_header=True, header_style="bold cyan", title="Keyboard Shortcuts")
    shortcut_table.add_column("Shortcut", style="cyan")
    shortcut_table.add_column("Action")
    shortcut_table.add_row("Tab", "Autocomplete commands/paths")
    shortcut_table.add_row("↑ / ↓", "Navigate command history")
    shortcut_table.add_row("Ctrl+C", "Cancel current input/generation")
    shortcut_table.add_row("Ctrl+D", "Exit aigent")
    shortcut_table.add_row("Ctrl+L", "Clear screen")
    shortcut_table.add_row(";;", "End multi-line input")

    console.print(Panel(Group(cmd_table, Text(), shortcut_table), title="[bold]Help[/bold]", border_style="blue"))


def display_tools():
    """Display available tools."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    for name, func in TOOL_REGISTRY.items():
        doc = func.__doc__ or ""
        first_line = doc.strip().split("\n")[0].strip()
        table.add_row(name, first_line)
    console.print(Panel(table, title="[bold]Available Tools[/bold]", border_style="blue"))


def fetch_ollama_models(base_url: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch available models from Ollama API."""
    # Convert OpenAI-compatible URL to Ollama native API
    ollama_url = base_url.replace("/v1", "").rstrip("/")

    try:
        # Get list of models
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("models", [])
    except Exception as e:
        console.print(f"[red]Could not fetch models: {e}[/red]")
        return None


def get_model_capabilities(model_info: Dict[str, Any], base_url: str) -> Dict[str, bool]:
    """Determine model capabilities based on model details."""
    model_name = model_info.get("name", "").lower()
    details = model_info.get("details", {})
    families = details.get("families", []) or []

    # Convert families to lowercase for comparison
    families_lower = [f.lower() for f in families]

    capabilities = {
        "vision": False,
        "tools": False,
        "thinking": False,
        "embedding": False,
        "code": False,
    }

    # Check for vision capability
    if "clip" in families_lower or "vision" in model_name or "llava" in model_name:
        capabilities["vision"] = True

    # Check for tool/function calling capability
    tool_models = ["qwen", "mistral", "llama3", "nemotron", "granite", "command-r", "firefunction"]
    if any(t in model_name for t in tool_models):
        capabilities["tools"] = True

    # Check for thinking/reasoning capability
    thinking_models = ["deepseek", "qwq", "thinking", "reason", "r1"]
    if any(t in model_name for t in thinking_models):
        capabilities["thinking"] = True

    # Check for embedding models
    if "embed" in model_name or "nomic" in model_name or "bge" in model_name:
        capabilities["embedding"] = True

    # Check for code-specialized models
    code_models = ["code", "starcoder", "codellama", "deepseek-coder", "codegemma", "qwen2.5-coder"]
    if any(c in model_name for c in code_models):
        capabilities["code"] = True

    return capabilities


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def display_models(base_url: str, current_model: str):
    """Display available Ollama models with their capabilities."""
    with console.status("[bold cyan]Fetching models...[/bold cyan]", spinner="dots"):
        models = fetch_ollama_models(base_url)

    if models is None:
        return

    if not models:
        console.print(Panel(
            "[yellow]No models installed.[/yellow]\n\n"
            "Install a model with: [cyan]ollama pull <model>[/cyan]",
            title="[bold]Models[/bold]",
            border_style="blue"
        ))
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="dim", justify="right")
    table.add_column("Vision", justify="center")
    table.add_column("Tools", justify="center")
    table.add_column("Thinking", justify="center")
    table.add_column("Code", justify="center")
    table.add_column("Embed", justify="center")

    for model in sorted(models, key=lambda x: x.get("name", "")):
        name = model.get("name", "unknown")
        size = model.get("size", 0)

        # Mark current model
        if name == current_model or name.split(":")[0] == current_model:
            name = f"[bold green]► {name}[/bold green]"

        caps = get_model_capabilities(model, base_url)

        def cap_icon(enabled: bool) -> str:
            return "[green]✓[/green]" if enabled else "[dim]–[/dim]"

        table.add_row(
            name,
            format_size(size),
            cap_icon(caps["vision"]),
            cap_icon(caps["tools"]),
            cap_icon(caps["thinking"]),
            cap_icon(caps["code"]),
            cap_icon(caps["embedding"]),
        )

    console.print(Panel(
        table,
        title="[bold]Available Models[/bold]",
        subtitle="[dim]Vision=image input | Tools=function calling | Thinking=reasoning | Code=programming | Embed=embeddings[/dim]",
        border_style="blue"
    ))


# =============================================================================
# INPUT HANDLING
# =============================================================================

# Available slash commands for completion
SLASH_COMMANDS = ["/help", "/clear", "/model", "/models", "/tools", "/project", "/exit", "/quit"]


class AigentCompleter:
    """Tab completion for aigent CLI."""

    def __init__(self):
        self.matches: List[str] = []

    def complete(self, text: str, state: int) -> Optional[str]:
        """Readline completion function."""
        if state == 0:
            # Get the full line buffer
            line = readline.get_line_buffer()
            self.matches = self._get_matches(line, text)

        if state < len(self.matches):
            return self.matches[state]
        return None

    def _get_matches(self, line: str, text: str) -> List[str]:
        """Get completion matches based on current input."""
        # Slash command completion
        if line.startswith("/"):
            return [cmd for cmd in SLASH_COMMANDS if cmd.startswith(text)]

        # File path completion (if text looks like a path)
        if text.startswith(("./", "../", "/", "~")) or "/" in text:
            return self._complete_path(text)

        # Also complete paths if it might be a file reference
        if text and not text.startswith("/"):
            path_matches = self._complete_path(text)
            if path_matches:
                return path_matches

        return []

    def _complete_path(self, text: str) -> List[str]:
        """Complete file paths."""
        try:
            # Expand ~ to home directory
            if text.startswith("~"):
                expanded = os.path.expanduser(text)
                prefix = "~"
                search_text = expanded
            else:
                prefix = ""
                search_text = text

            # Get directory and partial filename
            if os.path.isdir(search_text):
                directory = search_text
                partial = ""
            else:
                directory = os.path.dirname(search_text) or "."
                partial = os.path.basename(search_text)

            # List matching files
            matches = []
            if os.path.isdir(directory):
                for name in os.listdir(directory):
                    if name.startswith(partial):
                        full_path = os.path.join(directory, name)
                        # Add trailing slash for directories
                        if os.path.isdir(full_path):
                            name += "/"
                        # Reconstruct with original prefix
                        if prefix == "~":
                            home = os.path.expanduser("~")
                            result = "~" + full_path[len(home):]
                        else:
                            result = full_path
                        matches.append(result)

            return sorted(matches)
        except Exception:
            return []


# Global completer instance
completer = AigentCompleter()


# =============================================================================
# KEYBOARD SHORTCUTS & SIGNAL HANDLING
# =============================================================================

class InterruptHandler:
    """Handle keyboard interrupts gracefully."""

    def __init__(self):
        self.interrupted = False
        self.original_handler = None

    def __enter__(self):
        self.interrupted = False
        self.original_handler = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGINT, self.original_handler)

    def _handler(self, signum, frame):
        self.interrupted = True
        console.print("\n[yellow]⚡ Interrupted - finishing current operation...[/yellow]")


# Global interrupt handler
interrupt_handler = InterruptHandler()


def clear_screen():
    """Clear the terminal screen."""
    console.clear()
    display_welcome()
    display_status_bar()


def handle_ctrl_l():
    """Handle Ctrl+L to clear screen."""
    # This is called via readline binding
    clear_screen()
    return ""


def setup_keyboard_shortcuts():
    """Setup additional keyboard shortcuts via readline."""
    try:
        # Ctrl+L to clear screen
        # We need to use a custom approach since readline doesn't directly support this
        if "libedit" in readline.__doc__:
            # macOS
            readline.parse_and_bind("bind ^L ed-clear-screen")
        else:
            # Linux - GNU readline
            readline.parse_and_bind('"\\C-l": clear-screen')
    except Exception:
        pass


def setup_readline():
    """Setup readline for command history and tab completion."""
    try:
        # Load history
        if HISTORY_FILE.exists():
            readline.read_history_file(HISTORY_FILE)
        readline.set_history_length(1000)

        # Setup tab completion
        readline.set_completer(completer.complete)
        readline.set_completer_delims(" \t\n;")

        # Use tab for completion (handle different platforms)
        if "libedit" in readline.__doc__:
            # macOS uses libedit
            readline.parse_and_bind("bind ^I rl_complete")
            readline.parse_and_bind("bind ^L ed-clear-screen")
        else:
            # Linux uses GNU readline
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind('"\\C-l": clear-screen')

        # Setup keyboard shortcuts
        setup_keyboard_shortcuts()

    except Exception:
        pass


def save_readline():
    """Save readline history."""
    try:
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass


def get_multiline_input() -> Optional[str]:
    """Get input that can span multiple lines. End with ;; on a new line."""
    lines = []
    try:
        # Use standard input() for readline/tab-completion support
        # Print prompt with rich, then use input()
        console.print("[bold blue]You:[/bold blue] ", end="")
        first_line = input()
        if not first_line:
            return None
        lines.append(first_line)

        # Check for multi-line mode (if first line doesn't look complete)
        while True:
            if lines[-1].strip().endswith(";;"):
                # Remove the ;; marker
                lines[-1] = lines[-1].rsplit(";;", 1)[0]
                break
            if len(lines) == 1:
                # Single line input is fine
                break
            try:
                console.print("[dim]...[/dim] ", end="")
                continuation = input()
                if continuation.strip() == ";;":
                    break
                lines.append(continuation)
            except EOFError:
                break

    except EOFError:
        # Ctrl+D pressed - signal exit
        console.print()
        console.print("[dim]Ctrl+D pressed. Goodbye![/dim]")
        return "EXIT_SIGNAL"
    except KeyboardInterrupt:
        # Ctrl+C pressed - cancel current input
        console.print()
        console.print("[dim]Input cancelled.[/dim]")
        return None

    return "\n".join(lines).strip()


def handle_slash_command(command: str, model: str, base_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle slash commands.
    Returns (action, new_model) where:
    - action: None (not a command), 'exit', 'clear', 'continue'
    - new_model: New model name if switched, None otherwise
    """
    cmd_parts = command.strip().split(maxsplit=1)
    cmd = cmd_parts[0].lower()
    args = cmd_parts[1] if len(cmd_parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        console.print("[dim]Goodbye![/dim]")
        return "exit", None
    elif cmd == "/clear":
        console.print("[dim]Conversation cleared.[/dim]")
        return "clear", None
    elif cmd == "/help":
        display_help()
        return "continue", None
    elif cmd == "/model":
        if args:
            # Switch to new model
            new_model = args.strip()
            # Verify model exists by checking available models
            models = fetch_ollama_models(base_url)
            if models:
                available = [m.get("name", "").split(":")[0] for m in models]
                available_full = [m.get("name", "") for m in models]
                if new_model in available or new_model in available_full:
                    console.print(f"[green]✓[/green] Switched to model: [cyan]{new_model}[/cyan]")
                    return "continue", new_model
                else:
                    console.print(f"[red]✗[/red] Model not found: [cyan]{new_model}[/cyan]")
                    console.print(f"[dim]Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}[/dim]")
                    return "continue", None
            else:
                # Can't verify, just switch
                console.print(f"[yellow]![/yellow] Switching to model: [cyan]{new_model}[/cyan] (could not verify)")
                return "continue", new_model
        else:
            # Show current model
            console.print(f"[dim]Current model:[/dim] [cyan]{model}[/cyan]")
            console.print(f"[dim]Switch with:[/dim] /model <name>")
            return "continue", None
    elif cmd == "/models":
        display_models(base_url, model)
        return "continue", None
    elif cmd == "/tools":
        display_tools()
        return "continue", None
    elif cmd == "/project":
        return handle_project_command(args), None

    return None, None  # Not a slash command


# =============================================================================
# LLM INTERACTION
# =============================================================================

def create_client(base_url: str) -> OpenAI:
    """Create OpenAI client for Ollama."""
    return OpenAI(
        base_url=base_url,
        api_key="ollama"
    )


def display_connection_error(base_url: str, model: str, error: Exception):
    """Display a helpful error message for connection failures."""
    console.print()
    console.print(Panel(
        f"[bold red]Connection Failed[/bold red]\n\n"
        f"Could not connect to Ollama at [cyan]{base_url}[/cyan]\n\n"
        f"[bold]Possible causes:[/bold]\n"
        f"  1. Ollama is not running\n"
        f"  2. Wrong URL (check --url option)\n"
        f"  3. Model '[cyan]{model}[/cyan]' is not installed\n\n"
        f"[bold]Solutions:[/bold]\n"
        f"  • Start Ollama: [cyan]ollama serve[/cyan]\n"
        f"  • Pull model:   [cyan]ollama pull {model}[/cyan]\n"
        f"  • Check URL:    [cyan]aigent --url http://localhost:11434/v1[/cyan]\n\n"
        f"[dim]Error: {error}[/dim]",
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(1, 2)
    ))
    console.print()


def execute_llm_call_streaming(client: OpenAI, model: str, conversation: List[Dict[str, str]], base_url: str) -> Optional[str]:
    """Execute LLM call with streaming output."""
    global session
    full_response = ""
    streaming_display = StreamingDisplay(console)
    thinking_display = ThinkingDisplay()
    interrupted = False
    first_token_received = False
    stream = None
    stream_error = None

    # Use a thread to make the API call while showing thinking animation
    def make_api_call():
        nonlocal stream, stream_error
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=conversation,
                stream=True
            )
        except Exception as e:
            stream_error = e

    # Start API call in background thread
    api_thread = threading.Thread(target=make_api_call, daemon=True)
    api_thread.start()

    try:
        # Show thinking animation while waiting for API connection
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            # Animate while API call is being made
            while api_thread.is_alive():
                live.update(thinking_display.render())
                time.sleep(0.08)

            # Check for errors
            if stream_error:
                raise stream_error

            # Continue thinking animation while waiting for first token
            try:
                for chunk in stream:
                    # Keep animating until we get content
                    if not chunk.choices[0].delta.content:
                        live.update(thinking_display.render())
                        continue

                    content = chunk.choices[0].delta.content

                    # First token received - switch to streaming display
                    if not first_token_received:
                        first_token_received = True
                        streaming_display.start_time = time.time()

                    full_response += content

                    # Show streaming text with animated cursor
                    live.update(streaming_display.render_with_header(full_response))

            except KeyboardInterrupt:
                interrupted = True
                live.update(Text("⚡ Interrupted", style="bold yellow"))
                time.sleep(0.3)

            if not interrupted and full_response:
                # Show completion state briefly
                live.update(streaming_display.render_with_header(full_response, is_complete=True))
                time.sleep(0.3)

        # Track tokens and update connection status
        session.connected = True
        input_tokens = sum(estimate_tokens(m.get("content", "")) for m in conversation)
        output_tokens = estimate_tokens(full_response)
        session.add_tokens(input_tokens + output_tokens)

        if interrupted:
            # Return partial response with interruption note
            if full_response:
                full_response += "\n\n[Generation interrupted by user]"
            return full_response if full_response else None

        # Success sound
        play_sound("complete")

        return full_response
    except APIConnectionError as e:
        session.connected = False
        session.last_error = str(e)
        play_sound("error")
        display_connection_error(base_url, model, e)
        return None
    except APIStatusError as e:
        session.connected = False
        session.last_error = str(e)
        play_sound("error")
        display_connection_error(base_url, model, e)
        return None
    except Exception as e:
        session.connected = False
        session.last_error = str(e)
        play_sound("error")
        display_connection_error(base_url, model, e)
        return None


def execute_llm_call(client: OpenAI, model: str, conversation: List[Dict[str, str]], base_url: str, stream: bool = True) -> Optional[str]:
    """Execute LLM call."""
    global session
    try:
        if stream:
            return execute_llm_call_streaming(client, model, conversation, base_url)
        else:
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                response = client.chat.completions.create(
                    model=model,
                    messages=conversation,
                )

            result = response.choices[0].message.content

            # Track tokens
            session.connected = True
            input_tokens = sum(estimate_tokens(m.get("content", "")) for m in conversation)
            output_tokens = estimate_tokens(result)
            session.add_tokens(input_tokens + output_tokens)

            return result
    except APIConnectionError as e:
        session.connected = False
        session.last_error = str(e)
        display_connection_error(base_url, model, e)
        return None
    except APIStatusError as e:
        session.connected = False
        session.last_error = str(e)
        display_connection_error(base_url, model, e)
        return None
    except Exception as e:
        session.connected = False
        session.last_error = str(e)
        display_connection_error(base_url, model, e)
        return None


# =============================================================================
# CONFIRMATION DIALOGS
# =============================================================================

# Tools that always require confirmation
DANGEROUS_TOOLS = {
    "delete_file",
    "git_push",
    "git_reset",
}

# Dangerous command patterns for run_command
DANGEROUS_COMMAND_PATTERNS = [
    r'\brm\b',           # rm command
    r'\brm\s+-rf?\b',    # rm -r or rm -rf
    r'\bsudo\b',         # sudo
    r'\bchmod\b',        # chmod
    r'\bchown\b',        # chown
    r'\bmkfs\b',         # mkfs (format disk)
    r'\bdd\b',           # dd command
    r'\b>\s*/',          # redirect to root
    r'\bkill\b',         # kill process
    r'\bpkill\b',        # pkill
    r'\breboot\b',       # reboot
    r'\bshutdown\b',     # shutdown
    r'\bdropdb\b',       # drop database
    r'\bDROP\s+DATABASE\b',  # SQL drop database
    r'\bDROP\s+TABLE\b',     # SQL drop table
    r'\bTRUNCATE\b',         # SQL truncate
    r'\bDELETE\s+FROM\b',    # SQL delete
]


def is_dangerous_action(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if a tool action is dangerous and needs confirmation.
    Returns (is_dangerous, reason).
    """
    # Check dangerous tools
    if tool_name in DANGEROUS_TOOLS:
        if tool_name == "delete_file":
            path = args.get("path", "unknown")
            return True, f"Delete file/directory: {path}"
        elif tool_name == "git_push":
            remote = args.get("remote", "origin")
            branch = args.get("branch", "current branch")
            return True, f"Push to {remote}/{branch}"
        elif tool_name == "git_reset":
            mode = args.get("mode", "mixed")
            target = args.get("target", "HEAD")
            return True, f"Git reset --{mode} {target}"

    # Check run_command for dangerous patterns
    if tool_name == "run_command":
        command = args.get("command", "")
        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Potentially dangerous command: {command}"

    # Check sqlite_execute for destructive SQL
    if tool_name == "sqlite_execute":
        statement = args.get("statement", "").upper()
        if any(kw in statement for kw in ["DROP", "DELETE", "TRUNCATE"]):
            return True, f"Destructive SQL: {args.get('statement', '')[:50]}"

    return False, ""


def preview_file_edit(path: str, old_str: str, new_str: str) -> Tuple[bool, str]:
    """
    Generate a preview diff for a file edit.
    Returns (success, diff_text or error_message).
    """
    full_path = resolve_abs_path(path)

    # New file creation
    if old_str == "":
        preview_lines = [
            f"[bold green]+ Creating new file: {full_path}[/bold green]",
            "",
        ]
        # Show first 20 lines of new content
        new_lines = new_str.split('\n')[:20]
        for line in new_lines:
            preview_lines.append(f"[green]+ {line}[/green]")
        if len(new_str.split('\n')) > 20:
            preview_lines.append(f"[dim]... ({len(new_str.split(chr(10))) - 20} more lines)[/dim]")
        return True, '\n'.join(preview_lines)

    # Check if file exists
    if not full_path.exists():
        return False, f"File not found: {full_path}"

    try:
        original = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Error reading file: {e}"

    # Check if old_str exists
    if original.find(old_str) == -1:
        return False, f"String not found in file"

    # Generate the edited version
    edited = original.replace(old_str, new_str, 1)

    # Create unified diff
    original_lines = original.splitlines(keepends=True)
    edited_lines = edited.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        edited_lines,
        fromfile=f"a/{full_path.name}",
        tofile=f"b/{full_path.name}",
        lineterm=""
    )

    diff_text = ''.join(diff)

    if not diff_text:
        return False, "No changes detected"

    return True, diff_text


def display_edit_preview(path: str, old_str: str, new_str: str) -> bool:
    """
    Display a preview of file changes and ask for confirmation.
    Returns True if confirmed, False if cancelled.
    """
    success, result = preview_file_edit(path, old_str, new_str)

    console.print()

    if not success:
        console.print(Panel(
            f"[bold red]Preview Error[/bold red]\n\n{result}",
            border_style="red"
        ))
        return False

    # Display the diff with syntax highlighting
    console.print(Panel(
        Syntax(result, "diff", theme="monokai", line_numbers=False),
        title=f"[bold cyan]📝 File Edit Preview: {path}[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    ))

    # Ask for confirmation
    try:
        console.print("[bold cyan]Apply changes? [y/N]:[/bold cyan] ", end="")
        response = input().strip().lower()
        if response in ('y', 'yes', 'ja', 'j'):
            console.print("[green]✓ Changes applied[/green]")
            return True
        else:
            console.print("[yellow]✗ Changes discarded[/yellow]")
            return False
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]✗ Changes discarded[/yellow]")
        return False


def confirm_action(reason: str) -> bool:
    """
    Ask user to confirm a dangerous action.
    Returns True if confirmed, False if cancelled.
    """
    console.print()
    console.print(Panel(
        f"[bold yellow]⚠ Confirmation Required[/bold yellow]\n\n"
        f"{reason}\n\n"
        f"[dim]Press [bold]y[/bold] to confirm, [bold]n[/bold] to cancel[/dim]",
        border_style="yellow",
        padding=(0, 2)
    ))

    try:
        console.print("[bold yellow]Confirm? [y/N]:[/bold yellow] ", end="")
        response = input().strip().lower()
        if response in ('y', 'yes', 'ja', 'j'):
            console.print("[green]✓ Confirmed[/green]")
            return True
        else:
            console.print("[red]✗ Cancelled[/red]")
            return False
    except (EOFError, KeyboardInterrupt):
        console.print("\n[red]✗ Cancelled[/red]")
        return False


def execute_tool(name: str, args: Dict[str, Any], skip_confirm: bool = False) -> Dict[str, Any]:
    """Execute a tool and return the result."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}

    # Show preview for file edits
    if name == "edit_file" and not skip_confirm:
        path = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        if not display_edit_preview(path, old_str, new_str):
            return {"error": "Edit cancelled by user", "cancelled": True}

    # Check if action needs confirmation
    if not skip_confirm:
        is_dangerous, reason = is_dangerous_action(name, args)
        if is_dangerous:
            if not confirm_action(reason):
                return {"error": "Action cancelled by user", "cancelled": True}

    tool = TOOL_REGISTRY[name]
    try:
        sig = inspect.signature(tool)
        call_args = {}
        for param_name in sig.parameters:
            if param_name in args:
                call_args[param_name] = args[param_name]
        return tool(**call_args)
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# MAIN AGENT LOOP
# =============================================================================

def run_coding_agent_loop(
    model: str = "nemotron-3-nano",
    base_url: str = "http://localhost:11434/v1",
    stream: bool = True,
    single_command: Optional[str] = None,
    verbose: bool = False
):
    """Main agent loop."""
    global session, current_project, project_path

    # Initialize session state
    session = SessionState(
        model=model,
        base_url=base_url,
        connected=False,
        start_time=datetime.now()
    )

    # Detect and load project
    detected_path = detect_project()
    if detected_path:
        load_project(detected_path)
        if verbose and current_project:
            console.print(f"[dim]Project: {current_project.name}[/dim]")

    client = create_client(base_url)

    # Test connection
    try:
        urllib.request.urlopen(
            base_url.replace("/v1", "") + "/api/tags",
            timeout=5
        )
        session.connected = True
    except Exception:
        session.connected = False

    setup_readline()

    if verbose:
        console.print(f"[dim]Model: {model}[/dim]")
        console.print(f"[dim]Base URL: {base_url}[/dim]")
        console.print(f"[dim]Streaming: {stream}[/dim]")

    conversation = [{
        "role": "system",
        "content": get_full_system_prompt()
    }]

    # Single command mode
    if single_command:
        conversation.append({"role": "user", "content": single_command})
        process_agent_turn(client, model, base_url, conversation, stream, verbose)
        display_status_bar()
        return

    # Interactive mode
    display_welcome()
    display_status_bar()

    try:
        while True:
            user_input = get_multiline_input()

            # Handle exit signal (Ctrl+D)
            if user_input == "EXIT_SIGNAL":
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                result, new_model = handle_slash_command(user_input, model, base_url)
                if new_model:
                    model = new_model
                    session.model = new_model
                    display_status_bar()
                if result == "exit":
                    break
                elif result == "clear":
                    conversation = [{"role": "system", "content": get_full_system_prompt()}]
                    display_status_bar()
                    continue
                elif result == "continue":
                    continue

            conversation.append({"role": "user", "content": user_input})
            process_agent_turn(client, model, base_url, conversation, stream, verbose)
            display_status_bar()

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
    finally:
        save_readline()
        console.print("[dim]Session ended. Tokens used: [cyan]{:,}[/cyan][/dim]".format(session.total_tokens))


def process_agent_turn(
    client: OpenAI,
    model: str,
    base_url: str,
    conversation: List[Dict[str, str]],
    stream: bool,
    verbose: bool
):
    """Process a single agent turn (may involve multiple tool calls)."""

    while True:
        assistant_response = execute_llm_call(client, model, conversation, base_url, stream)

        # Handle connection errors
        if assistant_response is None:
            return

        tool_invocations = extract_tool_invocations(assistant_response)

        if not tool_invocations:
            # No tool calls - display response and break
            display_assistant_message(assistant_response)
            conversation.append({"role": "assistant", "content": assistant_response})
            break

        # Handle tool calls
        conversation.append({"role": "assistant", "content": assistant_response})

        for name, args in tool_invocations:
            display_tool_call(name, args)

            # Execute tool with animation
            with ToolExecutionDisplay(console, name):
                result = execute_tool(name, args)

            # Play sound based on result
            if result.get("error"):
                play_sound("error")
            else:
                play_sound("success")

            display_tool_result(result, name)

            conversation.append({
                "role": "user",
                "content": f"tool_result({json.dumps(result)})"
            })


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="aigent - AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode with defaults
  python main.py --model mistral          # Use mistral model
  python main.py -c "list all python files"  # Single command mode
  python main.py --no-stream              # Disable streaming output
        """
    )

    parser.add_argument(
        "-m", "--model",
        default=os.environ.get("OLLAMA_MODEL", "nemotron-3-nano"),
        help="Model to use (default: nemotron-3-nano or OLLAMA_MODEL env var)"
    )

    parser.add_argument(
        "-u", "--url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        help="Ollama API base URL (default: http://localhost:11434/v1)"
    )

    parser.add_argument(
        "-c", "--command",
        help="Execute a single command and exit"
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    run_coding_agent_loop(
        model=args.model,
        base_url=args.url,
        stream=not args.no_stream,
        single_command=args.command,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
