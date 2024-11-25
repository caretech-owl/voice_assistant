# Simple Voice Assistant

This is still WIP...

## Install

The project uses uv as packet manager. Make sure to install uv as according to the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Then change to this directory and simply call

```sh
uv run run.py
```

All dependencies are installed automatically. The first time the script is run, some model files will be downloaded, which may take some time. After setup, the assistant waits for "Hey Jarvis". The current configuration creates a German voice assistant.

## Development

To add another package to the requirements, simply run
```sh
uv add <package>
```
