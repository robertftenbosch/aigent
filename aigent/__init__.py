#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import re
import readline
import subprocess
import sys
import urllib.request
import urllib.error
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
from dataclasses import dataclass, field
from datetime import datetime

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


def display_status_bar():
    """Display the status bar at the bottom of the screen."""
    # Get terminal width
    width = console.width

    # Build status components
    model_part = f"[cyan]{session.model}[/cyan]"

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
        status_part = "[green]● connected[/green]"
    else:
        status_part = "[red]● disconnected[/red]"

    # Uptime
    uptime_part = f"[dim]{session.get_uptime()}[/dim]"

    # Build the status line
    parts = [model_part, cwd_part]
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

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks.
You have access to a series of tools you can execute. Here are the tools you can execute:

{tool_list_repr}

When you want to use a tool, reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally. Use markdown formatting in your responses.
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
    return SYSTEM_PROMPT.format(tool_list_repr=tool_str_repr)


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return list of (tool_name, args) requested in 'tool: name({...})' lines.
    """
    invocations = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("tool:"):
            continue
        try:
            after = line[len("tool:"):].strip()
            name, rest = after.split("(", 1)
            name = name.strip()
            if not rest.endswith(")"):
                continue
            json_str = rest[:-1].strip()
            args = json.loads(json_str)
            invocations.append((name, args))
        except Exception:
            continue
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

        # Use diff syntax highlighting for git diff
        if name == "git_diff" and result.get("stdout"):
            content = Syntax(result["stdout"][:3000], "diff", theme="monokai", line_numbers=False)
            console.print(Panel(
                content,
                title=f"[bold green]Git Diff[/bold green] [dim](exit: [{rc_style}]{rc}[/{rc_style}])[/dim]",
                border_style="green"
            ))
        else:
            console.print(Panel(
                Text.from_markup(output[:2000] + ("..." if len(output) > 2000 else "")),
                title=f"[bold green]Command Output[/bold green] [dim](exit: [{rc_style}]{rc}[/{rc_style}])[/dim]",
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
    """Display assistant message with markdown formatting."""
    console.print()
    console.print(Panel(
        Markdown(text),
        title="[bold yellow]Assistant[/bold yellow]",
        border_style="yellow",
        padding=(1, 2)
    ))
    console.print()


def display_welcome():
    """Display welcome message."""
    console.print()
    console.print(Panel(
        "[bold cyan]aigent[/bold cyan] - AI Coding Assistant\n\n"
        "[dim]Commands:[/dim]\n"
        "  [cyan]/help[/cyan]    - Show help\n"
        "  [cyan]/clear[/cyan]   - Clear conversation\n"
        "  [cyan]/model[/cyan]   - Show current model\n"
        "  [cyan]/exit[/cyan]    - Exit the agent\n\n"
        "[dim]Multi-line input:[/dim] End with [cyan];;[/cyan] on a new line",
        title="[bold]Welcome[/bold]",
        border_style="blue"
    ))
    console.print()


def display_help():
    """Display help information."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/model", "Show current model")
    table.add_row("/models", "List available Ollama models with capabilities")
    table.add_row("/tools", "List available tools")
    table.add_row("/exit, /quit", "Exit the agent")
    table.add_row(";;", "End multi-line input (on new line)")
    console.print(Panel(table, title="[bold]Help[/bold]", border_style="blue"))


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

def setup_readline():
    """Setup readline for command history."""
    try:
        if HISTORY_FILE.exists():
            readline.read_history_file(HISTORY_FILE)
        readline.set_history_length(1000)
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
        first_line = console.input("[bold blue]You:[/bold blue] ")
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
                continuation = console.input("[dim]...[/dim] ")
                if continuation.strip() == ";;":
                    break
                lines.append(continuation)
            except EOFError:
                break

    except (KeyboardInterrupt, EOFError):
        return None

    return "\n".join(lines).strip()


def handle_slash_command(command: str, model: str, base_url: str) -> Optional[str]:
    """Handle slash commands. Returns None to continue, 'exit' to exit, 'clear' to clear."""
    cmd = command.lower().strip()

    if cmd in ("/exit", "/quit"):
        console.print("[dim]Goodbye![/dim]")
        return "exit"
    elif cmd == "/clear":
        console.print("[dim]Conversation cleared.[/dim]")
        return "clear"
    elif cmd == "/help":
        display_help()
        return "continue"
    elif cmd == "/model":
        console.print(f"[dim]Current model:[/dim] [cyan]{model}[/cyan]")
        return "continue"
    elif cmd == "/models":
        display_models(base_url, model)
        return "continue"
    elif cmd == "/tools":
        display_tools()
        return "continue"

    return None  # Not a slash command


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

    try:
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            stream = client.chat.completions.create(
                model=model,
                messages=conversation,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Show streaming text
                    live.update(Text(full_response, style="yellow"))

        # Track tokens and update connection status
        session.connected = True
        input_tokens = sum(estimate_tokens(m.get("content", "")) for m in conversation)
        output_tokens = estimate_tokens(full_response)
        session.add_tokens(input_tokens + output_tokens)

        return full_response
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


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return the result."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}

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
    global session

    # Initialize session state
    session = SessionState(
        model=model,
        base_url=base_url,
        connected=False,
        start_time=datetime.now()
    )

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

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                result = handle_slash_command(user_input, model, base_url)
                if result == "exit":
                    break
                elif result == "clear":
                    conversation = [{"role": "system", "content": get_full_system_prompt()}]
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

            result = execute_tool(name, args)

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
