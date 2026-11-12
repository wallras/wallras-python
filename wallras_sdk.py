"""Wallras SDK: send a self-hosted model's internal signals to Wallras (inside-out mode).

No dependencies. Reports are sent in the background by default, so they never slow your app,
and a failure to report never raises into your code.

    from wallras_sdk import Wallras
    w = Wallras("http://localhost:8080", token="your WALLRAS_APP_TOKEN")

    # Hugging Face transformers
    out = model.generate(**inputs, max_new_tokens=256, output_logits=True, output_scores=True,
                         return_dict_in_generate=True)
    w.report_transformers(out, inputs["input_ids"].shape[1], tokenizer, model="llama-3.1-8b", prompt=prompt)

    # Any runtime that returns OpenAI-style logprobs (vLLM, llama.cpp server, Ollama...)
    w.report(model="llama-3.1-8b", prompt=prompt, answer=text,
             tokens=Wallras.tokens_from_openai(response["choices"][0]["logprobs"]["content"]))

    # Your own probe on hidden states (0 = the model doesn't believe it, 1 = it does)
    w.report(model="...", answer=text, probe_score=0.31)

    # A request that already went through the Wallras proxy: attach to it
    w.report(event_id=resp.headers["X-Wallras-Event-Id"], tokens=tokens)

    # Decision record: how a self-hosted model reached its answer, read from the network itself
    out = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True, output_logits=True,
                         output_hidden_states=True, output_attentions=True)   # load with attn_implementation="eager"
    w.explain_transformers(model, tokenizer, inputs, out, model_name="llama-3.1-8b", prompt=prompt)

    # Agent traces: report steps Wallras can't see (tool runs, lookups, API calls) into the
    # same run as your model calls (send X-Wallras-Trace: run_id on those calls too).
    with w.step(run_id, "search_flights", kind="api"):
        results = search(...)   # timed; an exception is recorded as a failed step and re-raised

    # Prompt management: fetch the version labelled production, with variables filled in. Send
    # p["header"] with the model call so the request is filed under that version.
    p = w.prompt("support-agent", customer="Asha", fallback={"messages": [...]})
    client.chat.completions.create(model=p["model"] or "gpt-5-mini", messages=p["messages"],
                                   extra_headers=p["header"])
"""
import contextlib
import json
import threading
import time
import urllib.request

__all__ = ["Wallras"]


class Wallras:
    def __init__(self, url="http://localhost:8080", token=None, timeout=5.0, background=True):
        if not str(url).lower().startswith(("http://", "https://")):
            raise ValueError("Wallras url must start with http:// or https://")   # never file:// or other schemes
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.background = background
        self._prompts = {}

    # ------------------------------------------------------------ prompts

    def prompt(self, name, label="production", version=None, cache_seconds=60, fallback=None, **variables):
        """The messages for a saved prompt, with variables filled in. Cached for cache_seconds. If Wallras
        can't be reached, the last good copy is used, then `fallback` (a dict with "messages"), so an
        outage never stops your app; with neither, it raises RuntimeError."""
        q = {"label": label} if version is None else {"version": str(int(version))}
        q.update({f"var.{k}": str(v) for k, v in variables.items()})
        key = (name, tuple(sorted(q.items())))
        hit = self._prompts.get(key)
        if hit and time.time() - hit[0] < cache_seconds:
            return hit[1]
        path = f"/wallras/prompts/{urllib.request.quote(str(name), safe='')}/resolve?" + \
            "&".join(f"{urllib.request.quote(k, safe='.')}={urllib.request.quote(v, safe='')}" for k, v in q.items())
        req = urllib.request.Request(f"{self.url}{path}",
                                     headers={**({"x-wallras-token": self.token} if self.token else {})})
        try:
            # Reviewed: only http(s) URLs reach here (checked in __init__).
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosemgrep # nosec B310
                got = json.loads(r.read())
            self._prompts[key] = (time.time(), got)
            return got
        except Exception as exc:
            if hit:
                return hit[1]
            if fallback is not None:
                return {"name": name, "version": None, "model": None, "params": {}, "header": {}, **fallback}
            raise RuntimeError(f"Wallras prompt {name!r} unavailable ({type(exc).__name__}) and no fallback given")

    # ------------------------------------------------------------ converters

    @staticmethod
    def tokens_from_openai(content):
        """OpenAI-style `logprobs.content` list -> Wallras token list."""
        out = []
        for t in content or []:
            if not isinstance(t, dict) or "logprob" not in t:
                continue
            alts = [x for x in t.get("top_logprobs") or [] if isinstance(x, dict) and "logprob" in x]
            out.append({"logprob": float(t["logprob"]), "text": t.get("token", ""),
                        "top": [float(x["logprob"]) for x in alts],
                        "alts": [[x.get("token", ""), float(x["logprob"])] for x in alts]})
        return out

    @staticmethod
    def tokens_from_steps(step_logprobs, chosen_ids, top_k=5):
        """Per generated token: the full log-probability vector (any sequence of floats) and the
        id that was chosen. Pure Python, so it works with lists from any framework."""
        out = []
        for row, tok in zip(step_logprobs, chosen_ids):
            row = list(row)
            top = sorted(row, reverse=True)[:top_k]
            out.append({"logprob": float(row[int(tok)]), "top": [float(x) for x in top]})
        return out

    def report_transformers(self, generate_output, prompt_length, tokenizer=None, top_k=5, **kw):
        """`model.generate(..., output_logits=True, return_dict_in_generate=True)` output, for
        decoder-only models, first sequence of the batch.

        Uses the raw logits when present (transformers 4.38+). `scores` are reshaped by sampling
        settings: temperature, top-k and top-p all change them, which would misstate how sure the
        model really was. Without raw logits, report with greedy decoding for accurate numbers."""
        seq = generate_output.sequences[0][prompt_length:]
        steps = getattr(generate_output, "logits", None) or generate_output.scores
        tokens = []
        for step, tok in zip(steps, seq):
            lp = step[0].float().log_softmax(-1)
            tokens.append({"logprob": float(lp[int(tok)]), "top": [float(x) for x in lp.topk(top_k).values.tolist()]})
        if tokenizer is not None and "answer" not in kw:
            kw["answer"] = tokenizer.decode(seq, skip_special_tokens=True)
        return self.report(tokens=tokens, **kw)

    def explain_transformers(self, model, tokenizer, inputs, generate_output, *, model_name=None,
                             prompt=None, event_id=None, top_k=5, layers=True, attention=True, **kw):
        """Read how a Hugging Face model reached its answer and send a decision record.

        Call model.generate(..., return_dict_in_generate=True, output_logits=True,
        output_hidden_states=True, output_attentions=True). Attention weights need the model
        loaded with attn_implementation="eager". Decoder-only models, first sequence of the batch.
        Per generated token it records: the token, its probability and the top alternatives
        (from raw logits, so sampling settings don't distort them), the layer from which the
        network kept it as its top choice (logit lens: each layer's hidden state through the
        final norm and output head), and the prompt tokens the last layer attended to most."""
        import torch  # only needed here
        ids = inputs["input_ids"]
        plen = ids.shape[1]
        seq = generate_output.sequences[0][plen:]
        steps = getattr(generate_output, "logits", None) or generate_output.scores
        hidden = getattr(generate_output, "hidden_states", None) if layers else None
        attns = getattr(generate_output, "attentions", None) if attention else None
        head = model.get_output_embeddings()
        norm = _final_norm(model)
        dec = lambda t: tokenizer.decode([int(t)])
        tokens, n_layers = [], None
        if not bool(torch.isfinite(steps[0][0]).any()):
            # Usually a half-precision model running on CPU: load it with torch_dtype=torch.float32.
            return {"recorded": False, "error": "the model's outputs are NaN or infinite; nothing to explain"}
        with torch.no_grad():
            for i, (step, tok) in enumerate(zip(steps, seq)):
                tok = int(tok)
                lp = step[0].float().log_softmax(-1)
                top = lp.topk(top_k)
                rec = {"text": dec(tok), "logprob": _finite(lp[tok]),
                       "top": [v for v in (_finite(x) for x in top.values.tolist()) if v is not None],
                       "alts": [[dec(j), _finite(v)] for v, j in zip(top.values.tolist(), top.indices.tolist())
                                if _finite(v) is not None]}
                if hidden is not None and i < len(hidden) and head is not None:
                    states = hidden[i][1:]          # skip the embedding layer
                    n_layers = len(states)
                    picks = []
                    for li, h in enumerate(states):
                        x = h[0, -1]
                        if norm is not None and li < n_layers - 1:  # the last layer is already normed
                            x = norm(x)
                        picks.append(int(head(x.to(head.weight.dtype)).argmax()))
                    settle = n_layers
                    for li in range(n_layers - 1, -1, -1):
                        if picks[li] != tok:
                            break
                        settle = li + 1
                    rec["layer"] = settle if picks[-1] == tok else n_layers
                if attns is not None and i < len(attns) and attns[i]:
                    a = attns[i][-1][0].float().mean(0)[-1][:plen]   # last layer, heads averaged
                    if bool(torch.isfinite(a).all()) and a.sum() > 0:
                        a = a / a.sum()
                        w, j = a.topk(min(3, plen))
                        rec["attn"] = [[int(x), float(y)] for x, y in zip(j.tolist(), w.tolist())]
                tokens.append(rec)
        prompt_tokens = [dec(t) for t in ids[0].tolist()] if attns is not None else None
        answer = tokenizer.decode(seq, skip_special_tokens=True)
        return self.report(tokens=tokens, model=model_name, answer=kw.pop("answer", answer),
                           prompt=prompt, event_id=event_id, prompt_tokens=prompt_tokens,
                           layers=n_layers, **kw)

    # ------------------------------------------------------------ sending

    def report(self, *, model=None, answer=None, prompt=None, event_id=None, tokens=None,
               probe_score=None, topic=None, prompt_tokens=None, layers=None, reasoning=None):
        body = {k: v for k, v in {"model": model, "answer": answer, "prompt": prompt, "event_id": event_id,
                                  "tokens": tokens or [], "probe_score": probe_score, "topic": topic,
                                  "prompt_tokens": prompt_tokens, "layers": layers,
                                  "reasoning": reasoning}.items()
                if v is not None}
        if self.background:
            threading.Thread(target=self._send, args=(body,), daemon=True).start()
            return None
        return self._send(body)

    @contextlib.contextmanager
    def step(self, trace_id, name, kind="tool"):
        """Time a step of an agent run and report it. Only the name, kind, timing, outcome and
        error type are sent, never inputs or outputs."""
        started, status, error = time.time(), "ok", None
        try:
            yield
        except BaseException as exc:
            status, error = "error", type(exc).__name__
            raise
        finally:
            body = {"name": str(name)[:80], "kind": kind, "started": started, "status": status,
                    "duration_ms": round((time.time() - started) * 1000, 1),
                    **({"error": error} if error else {})}
            path = f"/wallras/traces/{urllib.request.quote(str(trace_id), safe='')}/steps"
            if self.background:
                threading.Thread(target=self._send, args=(body, path), daemon=True).start()
            else:
                self._send(body, path)

    def _send(self, body, path="/wallras/internals"):
        try:  # never send NaN or infinity: the server would reject the whole report
            data = json.dumps(body, allow_nan=False).encode()
        except ValueError:
            return {"recorded": False, "error": "report contained NaN or infinite numbers"}
        req = urllib.request.Request(f"{self.url}{path}", data=data,
                                     headers={"content-type": "application/json",
                                              **({"x-wallras-token": self.token} if self.token else {})})
        try:
            # Reviewed: only http(s) URLs reach here (checked in __init__).
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosemgrep # nosec B310
                return json.loads(r.read() or b"{}")
        except Exception as exc:  # monitoring must never break your app
            return {"recorded": False, "error": type(exc).__name__}


def _final_norm(model):
    """The norm applied before the output head, for the logit lens. None if not found."""
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.decoder.final_layer_norm",
                 "transformer.norm_f", "backbone.norm_f"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _finite(x):
    """A plain float, or None for NaN and infinity (from broken or half-precision models)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None
