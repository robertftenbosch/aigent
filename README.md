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
   ollama pull llama3
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
| `-m, --model` | Model to use (default: llama3) |
| `-u, --url` | Ollama API URL (default: http://localhost:11434/v1) |
| `-c, --command` | Execute single command and exit |
| `--no-stream` | Disable streaming output |
| `-v, --verbose` | Enable verbose output |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear conversation |
| `/model` | Show current model |
| `/tools` | List available tools |
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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_MODEL` | Default model to use |
| `OLLAMA_BASE_URL` | Ollama API base URL |

## License

MIT
