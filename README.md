# Wallras clients for Python

Two single-file, zero-dependency clients for [Wallras](https://wallras-site.onrender.com), the AI monitoring proxy you run next to your app:

- **`wallras_mcp.py`**: an MCP server, so Claude Desktop, Claude Code, Cursor or your own agent can check drift, read requests and agent runs, work the review queue, fetch prompts and run regression sets. Read-only unless you pass `--enable-writes`.
- **`wallras_sdk.py`**: report what Wallras can't see on its own: a self-hosted model's token probabilities and decision records, agent steps (tool runs, lookups), and feedback; fetch prompt versions.

Both talk only to your own Wallras with your token. The Wallras engine itself ships as a Docker image (`ghcr.io/wallras/wallras-engine`); this repository holds the open-source clients.

## Install

```bash
pip install git+https://github.com/wallras/wallras-python
```

Or run the MCP server without installing anything:

```bash
uvx --from git+https://github.com/wallras/wallras-python wallras-mcp
```

Each file also works on its own: copy it next to your code.

## MCP server

```json
{"mcpServers": {"wallras": {
  "command": "uvx", "args": ["--from", "git+https://github.com/wallras/wallras-python", "wallras-mcp"],
  "env": {"WALLRAS_URL": "http://localhost:8080", "WALLRAS_TOKEN": "a viewer or admin token"}}}}
```

Add `"--enable-writes"` to `args` for the tools that change something (feedback, review decisions, saving or promoting prompts, regression runs, optionally graded by a `judge_model`). Those need an admin token; regression runs call your provider with `WALLRAS_PROVIDER_KEY` (and `WALLRAS_ANTHROPIC_KEY` / `WALLRAS_GEMINI_KEY`), which stay on your machine.

## SDK

```python
from wallras_sdk import Wallras
w = Wallras("http://localhost:8080", token="your WALLRAS_APP_TOKEN")
w.report(model="llama-3.1-8b", prompt=prompt, answer=text,
         tokens=Wallras.tokens_from_openai(response["choices"][0]["logprobs"]["content"]))
```

Reports go in the background and never raise into your code. See the [docs](https://wallras-site.onrender.com/docs.html) for traces, decision records and prompts.

## Try a hub with two nodes

[`examples/docker-compose.hub.yml`](examples/docker-compose.hub.yml) starts one hub and two nodes on your machine:

```bash
docker compose -f examples/docker-compose.hub.yml up -d
```

Send requests to a node (`http://localhost:8081/v1` or `:8082`, with your provider key), then open `http://localhost:8080/wallras/ui` and the Nodes tab.

## Try active-active

[`examples/docker-compose.active.yml`](examples/docker-compose.active.yml) starts Postgres, three Wallras containers and a load balancer. Any container answers any request, and together they keep one tamper-evident record:

```bash
docker compose -f examples/docker-compose.active.yml up -d
```

Send requests to `http://localhost:8080/v1`, then ask the MCP server's `wallras_fleet` tool (or `GET /wallras/fleet`) which containers answered.

## Issues

Bugs and ideas for the clients, or for Wallras itself: [open an issue](https://github.com/wallras/wallras-python/issues). Security reports: see [SECURITY.md](https://wallras-site.onrender.com/security.html), not public issues.

MIT licensed. The files here are copies of the ones published with each Wallras release.
