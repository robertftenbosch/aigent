# aigent

AI Coding Assistant powered by Ollama.

## Features

- 🤖 Agentic coding assistant with tool use
- 🎨 Beautiful CLI with rich output
- 🔧 20 built-in tools (file ops, shell, search, git)
- 🔀 Full git integration (status, diff, commit, branch, etc.)
- 📡 Streaming responses
- 💾 Command history
- 🔌 Works with any Ollama model

## Installation

### Global install (recommended)

```bash
# Install as global command
uv tool install git+https://github.com/robertftenbosch/aigent.git

# Or via SSH
uv tool install git+ssh://git@github.com/robertftenbosch/aigent.git

# Now use from anywhere
aigent
```

### As project dependency

```bash
# With uv (HTTPS)
uv add git+https://github.com/robertftenbosch/aigent.git

# With uv (SSH)
uv add git+ssh://git@github.com/robertftenbosch/aigent.git

# With pip (HTTPS)
pip install git+https://github.com/robertftenbosch/aigent.git

# With pip (SSH)
pip install git+ssh://git@github.com/robertftenbosch/aigent.git
```

### From source

```bash
git clone https://github.com/robertftenbosch/aigent.git
cd aigent
uv sync  # or: pip install -e .
```

## Prerequisites

1. Install [Ollama](https://ollama.ai)
2. Pull a model:
   ```bash
   ollama pull nemotron-3-nano
   ```
3. Start Ollama:
   ```bash
   ollama serve
   ```

## Usage

```bash
# Interactive mode
aigent

# With specific model
aigent --model mistral

# Single command
aigent -c "list all python files"

# Verbose mode
aigent -v
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-m, --model` | Model to use (default: nemotron-3-nano) |
| `-u, --url` | Ollama API URL (default: http://localhost:11434/v1) |
| `-c, --command` | Execute single command and exit |
| `--no-stream` | Disable streaming output |
| `-v, --verbose` | Enable verbose output |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/config` | Show/edit configuration |
| `/model` | Show or switch model |
| `/models` | List available models |
| `/project` | Project management |
| `/tools` | List available tools |
| `/clear` | Clear conversation |
| `/exit` | Exit aigent |

### Multi-line Input

End your input with `;;` on a new line for multi-line prompts.

## Available Tools

### File Operations

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `list_files` | List directory contents |
| `edit_file` | Edit or create files |
| `search_files` | Grep-like search in files |
| `find_files` | Find files by glob pattern |
| `create_directory` | Create directories |
| `delete_file` | Delete files or directories |

### System

| Tool | Description |
|------|-------------|
| `run_command` | Execute shell commands |
| `get_cwd` | Get current directory |

### Git

| Tool | Description |
|------|-------------|
| `git_status` | Show working tree status |
| `git_diff` | Show changes (staged/unstaged) |
| `git_log` | Show commit history |
| `git_add` | Stage files |
| `git_commit` | Commit changes |
| `git_branch` | List/manage branches |
| `git_checkout` | Switch branches |
| `git_pull` | Pull from remote |
| `git_push` | Push to remote |
| `git_stash` | Stash changes |
| `git_reset` | Reset HEAD |

### SQLite

| Tool | Description |
|------|-------------|
| `sqlite_query` | Execute SELECT query and return results |
| `sqlite_execute` | Execute INSERT/UPDATE/DELETE/CREATE statements |
| `sqlite_tables` | List all tables in a database |
| `sqlite_schema` | Show schema of database or table |

### Web

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using DuckDuckGo |
| `fetch_url` | Fetch and extract text from a URL |

### Python Packages

| Tool | Description |
|------|-------------|
| `pip_list` | List installed packages (optionally outdated only) |
| `pip_show` | Show info about an installed package |
| `pip_check` | Check for broken dependencies |
| `pypi_info` | Get package info from PyPI (online) |

### Image Analysis

| Tool | Description |
|------|-------------|
| `analyze_image` | Analyze image using vision model |

**Vision Model Support:**
- If current model has vision (e.g., llava, bakllava), it's used directly
- Otherwise, a dedicated vision model is used as an agent
- Default vision model: `llava` (configurable via `AIGENT_VISION_MODEL`)

## Configuration

Create a config file with `/config init` or manually at `~/.aigentrc`:

```json
{
  "language": "nl",
  "default_model": "nemotron-3-nano",
  "vision_model": "llava",
  "sounds": false,
  "show_tokens": true,
  "theme": "default"
}
```

| Setting | Description |
|---------|-------------|
| `language` | Response language: `nl` (Nederlands) or `en` (English) |
| `default_model` | Default Ollama model |
| `vision_model` | Model for image analysis |
| `sounds` | Enable/disable sounds |
| `show_tokens` | Show token counts |
| `custom_system_prompt` | Add custom instructions |

### Config Commands

```bash
/config              # Show current config
/config init         # Create default config file
/config set key val  # Change a setting
/config reload       # Reload from file
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_MODEL` | Default model (overrides config) |
| `OLLAMA_BASE_URL` | Ollama API base URL |
| `AIGENT_VISION_MODEL` | Vision model (overrides config) |
| `AIGENT_SOUNDS` | Enable sounds: 1=on (overrides config) |
| `AIGENT_LANGUAGE` | Language: nl/en (overrides config) |

## License

MIT
