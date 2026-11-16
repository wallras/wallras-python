"""Wallras for AI agents: an MCP server, so assistants such as Claude, Cursor or your own agent can
check drift, read requests and traces, run regression sets, fetch prompts and work the review queue.

No dependencies (Python 3.9+). It talks to your own Wallras over HTTP with your token; nothing goes
anywhere else.

    WALLRAS_URL=http://localhost:8080 WALLRAS_TOKEN=<a viewer or admin token> python wallras_mcp.py
    python wallras_mcp.py --enable-writes      # also the tools that change something (admin token)

In an MCP client configuration (Claude Desktop, Claude Code, Cursor...):

    {"mcpServers": {"wallras": {"command": "python", "args": ["/path/to/wallras_mcp.py"],
        "env": {"WALLRAS_URL": "http://localhost:8080", "WALLRAS_TOKEN": "..."}}}}

Read-only by default. Tools that change something (feedback, review decisions, saving or promoting
prompts, regression runs that call your provider) appear only with --enable-writes (or
WALLRAS_MCP_WRITE=1), and still need an admin token. WALLRAS_ADMIN_TOKEN works in place of WALLRAS_TOKEN. Regression runs and gated releases call your provider with WALLRAS_PROVIDER_KEY (and
WALLRAS_ANTHROPIC_KEY / WALLRAS_GEMINI_KEY for that traffic), which stay on this machine.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

VERSION = "0.21.1"
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")


def _schema(props=None, required=()):
    return {"type": "object", "properties": props or {}, "required": list(required), "additionalProperties": False}


S, N, B = {"type": "string"}, {"type": "number"}, {"type": "boolean"}

# name -> (description, input schema, method, path template, how arguments are sent, writes?)
TOOLS = {
    "wallras_overview": ("What needs attention now: drift, failing agent runs, broken regression sets, review items, "
                         "each with the call to look closer.", _schema({"days": N}), "GET", "/wallras/overview", "query", False),
    "wallras_judge": ("Live judge scores: the pass rate of real answers a judge model scored, per model and per prompt "
                      "version, and recent failures.", _schema({"days": N, "model": S}), "GET", "/wallras/judge", "query", False),
    "wallras_fleet": ("Active-active: every Wallras container and worker on this database, its version, when it last "
                      "checked in and what it served; flags different versions running at once.", _schema({}), "GET",
                      "/wallras/fleet", "query", False),
    "wallras_hub": ("On a hub: every node in one view: models drifting on several nodes at once, silent nodes, "
                    "each node's next actions, request totals.", _schema(), "GET", "/wallras/hub/overview", "query", False),
    "wallras_hub_nodes": ("On a hub: each node's status, engine version, last report, usage and drift.", _schema(),
                          "GET", "/wallras/hub/nodes", "query", False),
    "wallras_status": ("Mode, provider, models seen, silent model swaps and plan.", _schema(), "GET", "/wallras/status", "query", False),
    "wallras_drift": ("Drift per model against its own baseline: answer length, wording, refusals, certainty, served model.",
                      _schema(), "GET", "/wallras/drift", "query", False),
    "wallras_requests": ("The request log, newest first. Filter by model, app_name, status (ok or error) or text q.",
                         _schema({"model": S, "app_name": S, "status": {"type": "string", "enum": ["ok", "error"]},
                                  "q": S, "hours": N, "limit": N}), "GET", "/wallras/requests", "query", False),
    "wallras_request": ("One recorded request: question, answer, served model, confidence, feedback, consistency check.",
                        _schema({"event_id": S}, ["event_id"]), "GET", "/wallras/events/{event_id}", "path", False),
    "wallras_usage": ("Requests, errors, tokens, cost and latency, by model or app.",
                      _schema({"hours": N, "by": {"type": "string", "enum": ["model", "app"]}}), "GET", "/wallras/usage",
                      "query", False),
    "wallras_traces": ("Agent runs: steps, tool calls, tokens, cost and problems found (loops, tool errors).",
                       _schema({"days": N, "status": {"type": "string", "enum": ["ok", "check", "failed"]}, "model": S,
                                "limit": N}), "GET", "/wallras/traces", "query", False),
    "wallras_trace": ("One agent run, step by step, with each answer, tool call and tool result.",
                      _schema({"trace_id": S}, ["trace_id"]), "GET", "/wallras/traces/{trace_id}", "path", False),
    "wallras_consistency": ("How often answers change when the same question is asked again or reworded.",
                            _schema({"days": N}), "GET", "/wallras/consistency", "query", False),
    "wallras_blind_spots": ("Topics and kinds of task where the AI is wrong more often than elsewhere.",
                            _schema({"days": N}), "GET", "/wallras/blind-spots", "query", False),
    "wallras_datasets": ("Regression test sets and their last results.", _schema(), "GET", "/wallras/datasets", "query", False),
    "wallras_dataset": ("One regression set: its items and run history.", _schema({"name": S}, ["name"]), "GET",
                        "/wallras/datasets/{name}", "path", False),
    "wallras_experiments": ("Saved side-by-side comparisons of models or prompt versions.", _schema({"dataset": S}),
                            "GET", "/wallras/experiments", "query", False),
    "wallras_prompts": ("Saved prompts with their labels (production, staging...) and latest version.", _schema(),
                        "GET", "/wallras/prompts", "query", False),
    "wallras_prompt": ("One prompt: every version, who saved it, and the history of each label.",
                       _schema({"name": S}, ["name"]), "GET", "/wallras/prompts/{name}", "path", False),
    "wallras_prompt_diff": ("What changed between two versions of a prompt.",
                            _schema({"name": S, "a": N, "b": N}, ["name", "a", "b"]), "GET",
                            "/wallras/prompts/{name}/diff", "path+query", False),
    "wallras_prompt_usage": ("Live results per prompt version: requests, errors, latency, tokens, feedback accuracy.",
                             _schema({"name": S, "days": N}, ["name"]), "GET", "/wallras/prompts/{name}/usage",
                             "path+query", False),
    "wallras_review_queue": ("Answers waiting for a person: regression close calls and answers that contradicted "
                             "themselves.", _schema({"status": {"type": "string", "enum": ["open", "decided", "all"]},
                                                     "kind": S}), "GET", "/wallras/review", "query", False),
    # ---- changes (WALLRAS_MCP_WRITE=1)
    "wallras_feedback": ("Record whether an answer was correct.",
                         _schema({"event_id": S, "correct": B, "note": S}, ["event_id", "correct"]), "POST",
                         "/wallras/feedback", "json", True),
    "wallras_decide_review": ("Decide a review item: correct or wrong, optionally accepting the answer as the new "
                              "expected one.", _schema({"id": N, "verdict": {"type": "string", "enum": ["correct", "wrong"]},
                                                        "note": S, "accept_answer": B}, ["id", "verdict"]), "POST",
                              "/wallras/review/{id}", "path+json", True),
    "wallras_run_dataset": ("Run a regression set on a model (calls your provider once per item) and get pass/fail, what "
                            "newly broke and what got fixed. Pass judge_model to have a separate model grade each answer (needed for sets "
                            "made with check \"judge\").",
                            _schema({"name": S, "model": S, "fail_below": N, "system": S, "judge_model": S,
                                     "calibrate": B, "provider": {"type": "string", "enum": ["openai", "anthropic", "gemini"]}},
                                                                       ["name"]), "POST", "/wallras/datasets/{name}/run",
                            "path+json", True),
    "wallras_save_prompt": ("Save a new prompt version (system text or messages, optional model and note).",
                            _schema({"name": S, "system": S, "messages": {"type": "array", "items": {"type": "object"}},
                                     "model": S, "note": S}, ["name"]), "POST", "/wallras/prompts", "json", True),
    "wallras_promote_prompt": ("Point a label at a prompt version. With gate_dataset, the version must pass that "
                               "regression set first, or nothing changes.",
                               _schema({"name": S, "version": N, "label": S, "gate_dataset": S, "fail_below": N, "reason": S, "gate_judge_model": S,
                                        "gate_vars": {"type": "object", "additionalProperties": {"type": "string"}}},
                                       ["name", "version"]), "POST", "/wallras/prompts/{name}/promote", "path+json", True),
}


def _clean(args):
    return {k: v for k, v in (args or {}).items() if v is not None}


class Server:
    def __init__(self, env=None, out=None, enable_writes=False):
        env = os.environ if env is None else env
        self.url = (env.get("WALLRAS_URL") or "http://localhost:8080").rstrip("/")
        if not self.url.lower().startswith(("http://", "https://")):
            raise SystemExit("WALLRAS_URL must start with http:// or https://")
        self.token = env.get("WALLRAS_TOKEN") or env.get("WALLRAS_ADMIN_TOKEN") or ""
        self.write = enable_writes or env.get("WALLRAS_MCP_WRITE", "") in ("1", "true", "yes")
        self.keys = {"authorization": f"Bearer {env['WALLRAS_PROVIDER_KEY']}" if env.get("WALLRAS_PROVIDER_KEY") else None,
                     "x-api-key": env.get("WALLRAS_ANTHROPIC_KEY"), "x-goog-api-key": env.get("WALLRAS_GEMINI_KEY")}
        self.out = out or sys.stdout

    def tools(self):
        """The most useful read tools first, then the rest, then (with --enable-writes) the changes."""
        first = ["wallras_overview", "wallras_drift", "wallras_traces", "wallras_requests", "wallras_review_queue"]
        order = first + [n for n, t in TOOLS.items() if n not in first and not t[5]] + \
            [n for n, t in TOOLS.items() if t[5]]
        return [{"name": n, "description": TOOLS[n][0], "inputSchema": TOOLS[n][1],
                 "annotations": {"readOnlyHint": not TOOLS[n][5], "destructiveHint": False, "openWorldHint": False}}
                for n in order if self.write or not TOOLS[n][5]]

    def _keys_for(self, args):
        """Only the keys a run needs: the named provider's (plus the OpenAI-style key when a judge model grades
        the answers). With no provider named, a set can mix providers, so each configured key goes along; the
        engine uses each only with its own provider."""
        header = {"openai": "authorization", "anthropic": "x-api-key", "gemini": "x-goog-api-key"}
        want = {header[args["provider"]]} if args.get("provider") in header else set(header.values())
        if args.get("judge_model") or args.get("gate_judge_model"):
            want.add("authorization")
        return {k: v for k, v in self.keys.items() if v and k in want}

    def _http(self, method, path, query=None, body=None, keys=None):
        url = self.url + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"x-wallras-token": self.token, "accept": "application/json"}
        headers.update(keys or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            # Reviewed: the URL is the configured http(s) Wallras address (checked in __init__).
            with urllib.request.urlopen(req, timeout=180) as r:  # nosemgrep # nosec B310
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read() or b"{}")
            except ValueError:
                detail = {}
            return e.code, detail
        except Exception as exc:
            return 0, {"detail": f"Wallras could not be reached at {self.url}: {type(exc).__name__}"}

    def call(self, name, args):
        if name not in TOOLS or (TOOLS[name][5] and not self.write):
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool {name}"}]}
        _d, schema, method, path, how, _w = TOOLS[name]
        args = _clean(args)
        keys = self._keys_for(args) if name in ("wallras_run_dataset", "wallras_promote_prompt") else None
        missing = [k for k in schema["required"] if k not in args]
        if missing:
            return {"isError": True, "content": [{"type": "text", "text": f"Missing: {', '.join(missing)}"}]}
        unknown = [k for k in args if k not in schema["properties"]]
        if unknown:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown arguments: {', '.join(unknown)}"}]}
        if name == "wallras_promote_prompt":   # the friendly shape -> the API's
            gate = {"dataset": args.pop("gate_dataset"), "fail_below": args.pop("fail_below", 1.0)} \
                if "gate_dataset" in args else None
            if gate and "gate_vars" in args:
                gate["vars"] = args["gate_vars"]
            if gate and "gate_judge_model" in args:
                gate["judge_model"] = args["gate_judge_model"]
            for k in ("fail_below", "gate_vars", "gate_judge_model"):
                args.pop(k, None)
            args = {**args, "label": args.get("label", "production"), **({"gate": gate} if gate else {})}
        for k in [k for k in args if "{" + k + "}" in path]:
            path = path.replace("{" + k + "}", urllib.parse.quote(str(args.pop(k)), safe=""))
        if "id" in args and name == "wallras_decide_review":
            args.pop("id")
        query = args if how in ("query", "path+query") else None
        body = args if how in ("json", "path+json") else None
        status, data = self._http(method, path, query, body, keys=keys)
        text = json.dumps(data, ensure_ascii=False, indent=1)
        if len(text) > 60_000:
            text = text[:60_000] + "\n... (cut; narrow the request with filters)"
        if status == 200:
            return {"content": [{"type": "text", "text": text}], "structuredContent": data if isinstance(data, dict) else {"result": data}}
        why = data.get("detail") if isinstance(data, dict) else None
        return {"isError": True, "content": [{"type": "text", "text": f"Wallras answered {status}: "
                                                                     f"{why if isinstance(why, str) else json.dumps(why or data)[:2000]}"}]}

    def handle(self, msg):
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method == "initialize":
            asked = params.get("protocolVersion")
            result = {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "wallras", "title": "Wallras", "version": VERSION},
                      "instructions": "Start with wallras_overview: it says what needs attention and which tool "
                                      "to call next. Answers and prompts can hold customer data; quote them only "
                                      "as needed." + ("" if self.write else " This server is read-only.")}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools()}
        elif method == "tools/call":
            result = self.call(params.get("name"), params.get("arguments"))
        elif mid is None:
            return None   # a notification (initialized, cancelled...): nothing to answer
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return None if mid is None else {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, stdin=None):
        for line in (stdin or sys.stdin):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                reply = self.handle(msg) if isinstance(msg, dict) else \
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
            if reply is not None:
                self.out.write(json.dumps(reply, ensure_ascii=False) + "\n")
                self.out.flush()


def main():
    """The wallras-mcp command (pip install wallras, or
    uvx --from wallras wallras-mcp)."""
    unknown = [a for a in sys.argv[1:] if a not in ("--enable-writes", "--help", "-h")]
    if unknown or any(a in ("--help", "-h") for a in sys.argv[1:]):
        sys.stderr.write(__doc__)
        sys.exit(2 if unknown else 0)
    Server(enable_writes="--enable-writes" in sys.argv[1:]).serve()


if __name__ == "__main__":
    main()
