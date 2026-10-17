#!/usr/bin/env python3
"""JitRL：受限候选动作的 Gemini 评分与本地经验回放。"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_SCORE_TOKENS = 64000
DEFAULTS = {
    "base_url": "http://localhost:8787", "model": "gemini-3.8-flash-medium",
    "api_key_env": "JITRL_API_KEY", "api_key_fallback": "local-proxy",
    "data_dir": "~/.codex/jitrl-data", "beta": 1.0,
    "gamma": .9, "k": 10, "threshold": .2, "exploration_rate": .05,
    "exploration_bonus": 1, "epsilon": 1e-8, "timeout": 45,
    "max_tokens": MAX_SCORE_TOKENS,
}

class JitRLError(Exception):
    pass

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def tokens(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", text.lower()))

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise JitRLError(f"读取 JSON 失败：{path}：{exc}") from exc

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def load_config(path):
    cfg = dict(DEFAULTS)
    if path.exists():
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise JitRLError("配置必须是 JSON 对象")
        cfg.update(raw)
    cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser())
    for key in ("beta", "gamma", "threshold", "exploration_rate", "exploration_bonus", "epsilon", "timeout"):
        if not finite(cfg.get(key)):
            raise JitRLError(f"配置字段必须是有限数：{key}")
    if not isinstance(cfg.get("k"), int) or cfg["k"] <= 0:
        raise JitRLError("配置 k 必须是正整数")
    if type(cfg.get("max_tokens")) is not int or not 1 <= cfg["max_tokens"] <= MAX_SCORE_TOKENS:
        raise JitRLError(f"配置 max_tokens 必须是 1 到 {MAX_SCORE_TOKENS} 的整数")
    if not 0 <= cfg["gamma"] <= 1 or not 0 <= cfg["threshold"] <= 1 or not 0 <= cfg["exploration_rate"] <= 1 or cfg["epsilon"] <= 0 or cfg["timeout"] <= 0:
        raise JitRLError("配置 gamma/threshold/exploration_rate/epsilon/timeout 越界")
    return cfg

def validate_task(task):
    if not isinstance(task, dict):
        raise JitRLError("任务必须是 JSON 对象")
    for key in ("task_type", "environment_version", "state", "goal", "candidates"):
        if key not in task:
            raise JitRLError(f"任务缺少字段：{key}")
    if not isinstance(task["candidates"], list) or not task["candidates"]:
        raise JitRLError("candidates 必须是非空数组")
    seen = set()
    for c in task["candidates"]:
        if not isinstance(c, dict) or not all(k in c for k in ("id", "description", "tool", "args")):
            raise JitRLError("每个候选动作必须包含 id、description、tool、args")
        if not isinstance(c["id"], str) or c["id"] in seen:
            raise JitRLError("候选动作 id 必须是唯一字符串")
        seen.add(c["id"])
    return task

def action_key(action):
    return canonical({"tool": action.get("tool"), "args": action.get("args")})

def response_text(value):
    """提取 /v1/messages 的文本段，兼容常见 OpenAI/代理响应形状。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(response_text(x) for x in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        for key in ("content", "output", "choices", "message"):
            if key in value:
                out = response_text(value[key])
                if out:
                    return out
    return ""

def validate_score_map(scores, ids):
    if not isinstance(scores, dict) or set(scores) != set(ids):
        raise JitRLError("模型评分必须完整覆盖所有候选动作")
    for cid in ids:
        if not finite(scores[cid]) or not 0 <= float(scores[cid]) <= 100:
            raise JitRLError("模型评分 confidence 必须是 0 到 100 的有限数")
    return {cid: float(scores[cid]) for cid in ids}

def parse_scores(text, ids):
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        obj = json.loads(clean)
    except Exception as exc:
        raise JitRLError(f"模型评分不是合法 JSON：{exc}") from exc
    rows = obj.get("scores") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise JitRLError("模型评分必须完整覆盖所有候选动作")
    got = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not finite(row.get("confidence")):
            raise JitRLError("模型评分字段无效")
        if row["id"] in got or row["id"] not in ids or not 0 <= float(row["confidence"]) <= 100:
            raise JitRLError("模型评分 id 重复、未知或 confidence 越界")
        got[row["id"]] = float(row["confidence"])
    return validate_score_map(got, ids)

class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self.key = os.environ.get(cfg["api_key_env"], cfg.get("api_key_fallback", ""))

    def call(self, path, body=None):
        url = self.cfg["base_url"].rstrip("/") + path
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers["anthropic-version"] = "2023-06-01"
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
            headers["x-api-key"] = self.key
        req = Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers,
                      method="GET" if body is None else "POST")
        try:
            with urlopen(req, timeout=float(self.cfg["timeout"])) as res:
                raw = res.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise JitRLError(f"API 请求失败：{exc}") from exc
        try:
            return json.loads(raw)
        except Exception as exc:
            raise JitRLError(f"API 响应不是 JSON：{exc}") from exc

    def score(self, task):
        """向 Gemini 请求 task 中全部候选的 0—100 评分，返回经校验的映射和原始文本。"""
        ids = [c["id"] for c in task["candidates"]]
        prompt = {"task_type": task["task_type"], "environment_version": task["environment_version"],
                  "state": task["state"], "goal": task["goal"], "candidates": task["candidates"],
                  "response_format": {"scores": [{"id": "candidate id", "confidence": "0..100"}]}}
        body = {"model": self.cfg["model"], "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                "max_tokens": self.cfg.get("max_tokens", MAX_SCORE_TOKENS)}
        last = None
        for attempt in range(2):
            try:
                raw = self.call("/v1/messages", body)
                text = response_text(raw)
                # fixed by gyc 2026-10-17：评分：先识别预算截断，再解析 JSON；不补括号伪造完整评分。
                if raw.get("stop_reason") == "max_tokens":
                    last = JitRLError(f"模型评分被截断：stop_reason=max_tokens，预算={body['max_tokens']}，文本字符数={len(text)}")
                    if attempt == 0 and body["max_tokens"] < MAX_SCORE_TOKENS:
                        body["max_tokens"] = min(body["max_tokens"] * 2, MAX_SCORE_TOKENS)
                        continue
                    break
                scores = parse_scores(text, ids)
                return scores, text
            except JitRLError as exc:
                last = exc
                if attempt == 0:
                    body["messages"].append({"role": "user", "content": "仅返回合法 JSON，字段为 scores 数组，每项包含 id 和 0 到 100 的 confidence。"})
        raise last

class Store:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg["data_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / "jitrl.sqlite3"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, episode_id TEXT, step_index INTEGER,
          task_json TEXT NOT NULL, task_type TEXT, env TEXT, state_json TEXT, goal TEXT, candidates_json TEXT,
          z_json TEXT, prob_json TEXT, value REAL, q_json TEXT, advantage_json TEXT, normalized_json TEXT,
          selected_json TEXT, reason TEXT, model TEXT, response TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS experiences(id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT, decision_id TEXT UNIQUE,
          task_type TEXT, env TEXT, state_json TEXT, goal TEXT, action_json TEXT, reward REAL, return REAL);
        CREATE TABLE IF NOT EXISTS episodes(episode_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, finished INTEGER, created_at REAL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def decide(self, task, scores, response, seed=None):
        """按任务状态检索回报并采样动作；返回含决策 ID、价值与概率的凭据，不执行工具。"""
        validate_task(task)
        episode = task.get("episode_id") or str(uuid.uuid4())
        if self.db.execute("SELECT 1 FROM episodes WHERE episode_id=?", (episode,)).fetchone():
            raise JitRLError("episode 已经 finalized，不能继续 decide")
        ids = [c["id"] for c in task["candidates"]]
        scores = validate_score_map(scores, ids)
        next_step = self.db.execute("SELECT COALESCE(MAX(step_index)+1, 0) FROM decisions WHERE episode_id=?", (episode,)).fetchone()[0]
        task = dict(task, episode_id=episode, step_index=int(next_step))
        rows = self.db.execute("SELECT * FROM experiences WHERE task_type=? AND env=? ORDER BY id DESC LIMIT 5000",
                               (task["task_type"], task["environment_version"])).fetchall()
        target = tokens({"state": task["state"], "goal": task["goal"]})
        ranked = [(r, jaccard(target, tokens({"state": json.loads(r["state_json"]), "goal": r["goal"]}))) for r in rows]
        ranked = [r for r, similarity in ranked if similarity >= float(self.cfg["threshold"])]
        ranked = sorted(ranked, key=lambda r: jaccard(target, tokens({"state": json.loads(r["state_json"]), "goal": r["goal"]})), reverse=True)[:int(self.cfg["k"])]
        # added by gyc 2026-10-17：优势：仅在当前合法候选集内匹配历史动作，避免导入动作绕过合法性边界。
        V = sum(float(r["return"]) for r in ranked) / len(ranked) if ranked else 0.0
        rng = random.Random(seed)
        q, counts = {}, {}
        for c in task["candidates"]:
            key = action_key(c)
            matches = [r for r in ranked if r["action_json"] == key]
            counts[c["id"]] = len(matches)
            q[c["id"]] = (sum(float(r["return"]) for r in matches) / len(matches)) if matches else 0.0
            # fixed by gyc 2026-10-17：探索：概率 λ 只控制未见动作的乐观估值，动作始终按最终 Softmax 采样。
            if not matches and ranked and rng.random() < float(self.cfg["exploration_rate"]):
                q[c["id"]] = V + float(self.cfg["exploration_bonus"]) / len(ranked)
        adv = {cid: q[cid] - V for cid in q}
        scale = max((abs(x) for x in adv.values()), default=0.0)
        an = {cid: x / (scale + float(self.cfg["epsilon"])) for cid, x in adv.items()}
        total = sum(scores.values()) + float(self.cfg["epsilon"]) * len(scores)
        # added by gyc 2026-10-17：概率：把 0 到 100 的口头 confidence 映射为归一化先验的 log 概率。
        z = {c["id"]: math.log((scores[c["id"]] + float(self.cfg["epsilon"])) / total) for c in task["candidates"]}
        logits = {cid: z[cid] + float(self.cfg["beta"]) * an[cid] for cid in z}
        peak = max(logits.values())
        expv = {cid: math.exp(logits[cid] - peak) for cid in logits}
        denom = sum(expv.values())
        probs = {cid: expv[cid] / denom for cid in expv}
        x, selected_id = rng.random(), next(iter(probs))
        for cid, p in probs.items():
            x -= p
            if x <= 0:
                selected_id = cid
                break
        reason = "cold start" if not ranked else "advantage weighted"
        selected = next(c for c in task["candidates"] if c["id"] == selected_id)
        did = str(uuid.uuid4())
        rec = {"decision_id": did, "episode_id": episode, "step_index": int(task.get("step_index", 0)), "task": task,
               "z": z, "probabilities": probs, "V": V, "Q": q, "advantage": adv, "normalized_advantage": an,
               "matched": [{"episode_id": r["episode_id"], "decision_id": r["decision_id"]} for r in ranked],
               "selected": selected, "reason": reason, "model": self.cfg["model"], "response": response}
        self.db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (did, episode, rec["step_index"],
            canonical(task), task["task_type"], task["environment_version"], canonical(task["state"]), str(task["goal"]),
            canonical(task["candidates"]), canonical(z), canonical(probs), V, canonical(q), canonical(adv), canonical(an),
            canonical(selected), reason, self.cfg["model"], response, time.time()))
        self.db.commit()
        return rec

    def finish(self, feedback):
        """依据实际执行证据和逐步奖励，反向计算回报并原子保存一个完整回合。"""
        if (not isinstance(feedback, dict) or not isinstance(feedback.get("steps"), list)
                or not feedback.get("steps") or not feedback.get("episode_id")):
            raise JitRLError("feedback 必须包含 episode_id 和非空 steps")
        eid, steps = feedback["episode_id"], feedback["steps"]
        if self.db.execute("SELECT 1 FROM episodes WHERE episode_id=?", (eid,)).fetchone():
            raise JitRLError("episode 已经 finalized")
        seen, records, previous_step = set(), [], -1
        for i, item in enumerate(steps):
            if not isinstance(item, dict) or not item.get("decision_id") or item["decision_id"] in seen:
                raise JitRLError("feedback decision_id 必须唯一")
            if (not item.get("evidence") or not isinstance(item["evidence"], str) or not item["evidence"].strip()
                    or not item.get("reward_source") or not isinstance(item["reward_source"], str) or not item["reward_source"].strip()
                    or not finite(item.get("reward"))):
                raise JitRLError("feedback evidence/reward_source 非空且 reward 必须为有限数")
            row = self.db.execute("SELECT * FROM decisions WHERE decision_id=? AND episode_id=?", (item["decision_id"], eid)).fetchone()
            if not row:
                raise JitRLError("feedback 包含未知或跨 episode 的 decision_id")
            if int(row["step_index"]) <= previous_step:
                raise JitRLError("feedback steps 必须按 step_index 严格升序")
            previous_step = int(row["step_index"])
            if not -1 <= float(item["reward"]) <= 1:
                raise JitRLError("reward 必须位于 [-1, 1]")
            seen.add(item["decision_id"]); records.append((row, item))
        # fixed by gyc 2026-10-17：回报：缺失中间或末尾决策会截断回报，必须先补齐全回合反馈。
        expected = {r[0] for r in self.db.execute("SELECT decision_id FROM decisions WHERE episode_id=?", (eid,))}
        if seen != expected:
            raise JitRLError("feedback 必须覆盖该回合全部决策，不能遗漏步骤")
        returns = [0.0] * len(records); running = 0.0
        for i in range(len(records) - 1, -1, -1):
            running = float(records[i][1]["reward"]) + float(self.cfg["gamma"]) * running
            returns[i] = running
        payload = {"episode_id": eid, "outcome": feedback.get("outcome", ""), "steps": []}
        with self.db:
            for (row, item), ret in zip(records, returns):
                task = json.loads(row["task_json"])
                decision_record = {"decision_id": row["decision_id"], "episode_id": eid, "step_index": row["step_index"],
                                   "task": task, "z": json.loads(row["z_json"]), "probabilities": json.loads(row["prob_json"]),
                                   "V": row["value"], "Q": json.loads(row["q_json"]), "advantage": json.loads(row["advantage_json"]),
                                   "normalized_advantage": json.loads(row["normalized_json"]), "selected": json.loads(row["selected_json"]),
                                   "reason": row["reason"], "model": row["model"], "response": row["response"]}
                payload["steps"].append({"decision": decision_record, "decision_id": row["decision_id"],
                                         "selected": json.loads(row["selected_json"]), "evidence": item["evidence"],
                                         "reward": float(item["reward"]), "reward_source": item.get("reward_source", ""), "return": ret})
                self.db.execute("INSERT INTO experiences(episode_id,decision_id,task_type,env,state_json,goal,action_json,reward,return) VALUES(?,?,?,?,?,?,?,?,?)",
                    (eid, row["decision_id"], row["task_type"], row["env"], row["state_json"], row["goal"], action_key(json.loads(row["selected_json"])), float(item["reward"]), ret))
            self.db.execute("INSERT INTO episodes VALUES(?,?,?,?)", (eid, canonical(payload), 1, time.time()))
        return {"episode_id": eid, "count": len(records), "returns": returns}

    def export(self, eid):
        row = self.db.execute("SELECT payload_json FROM episodes WHERE episode_id=?", (eid,)).fetchone()
        if not row:
            raise JitRLError("episode 不存在或尚未 finalized")
        payload = json.loads(row["payload_json"])
        return {"schema_version": 1, "episode": payload,
                "sha256": hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()}

    def import_bundle(self, bundle):
        """校验导出包的摘要与结构，去重恢复经验；不把包中动作当作执行指令。"""
        if not isinstance(bundle, dict) or bundle.get("schema_version") != 1 or not isinstance(bundle.get("episode"), dict):
            raise JitRLError("bundle schema_version 或 episode 无效")
        payload = bundle["episode"]
        digest = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
        if bundle.get("sha256") != digest:
            raise JitRLError("bundle sha256 校验失败")
        eid = payload.get("episode_id")
        if not isinstance(eid, str) or not isinstance(payload.get("steps"), list):
            raise JitRLError("bundle episode 字段无效")
        old = self.db.execute("SELECT payload_json FROM episodes WHERE episode_id=?", (eid,)).fetchone()
        if old:
            if old["payload_json"] == canonical(payload):
                return {"episode_id": eid, "status": "idempotent"}
            raise JitRLError("同 episode_id 的 bundle 内容冲突")
        prepared, seen = [], set()
        for step in payload["steps"]:
            if (not isinstance(step, dict) or not isinstance(step.get("decision"), dict)
                    or not isinstance(step.get("decision_id"), str) or not step["decision_id"]
                    or step["decision_id"] in seen or not step.get("evidence")
                    or not isinstance(step.get("evidence"), str) or not step["evidence"].strip() or not step.get("reward_source")
                    or not isinstance(step.get("reward_source"), str) or not step["reward_source"].strip() or not finite(step.get("return"))
                    or not finite(step.get("reward")) or not -1 <= float(step.get("reward")) <= 1):
                raise JitRLError("bundle step 字段无效")
            d = step["decision"]
            task = d.get("task", d)
            validate_task(task)
            if d.get("decision_id") != step["decision_id"] or d.get("episode_id") != eid or task.get("episode_id") != eid:
                raise JitRLError("bundle decision 与 episode 不一致")
            selected = step.get("selected")
            if not isinstance(selected, dict) or action_key(selected) not in {action_key(c) for c in task["candidates"]}:
                raise JitRLError("bundle selected 不属于当前合法候选")
            if not isinstance(d.get("selected"), dict) or action_key(d["selected"]) != action_key(selected):
                raise JitRLError("bundle decision selected 与 step 不一致")
            probs = d.get("probabilities")
            if not isinstance(probs, dict) or set(probs) != {c["id"] for c in task["candidates"]} or not all(finite(v) and 0 <= float(v) <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
                raise JitRLError("bundle probabilities 无效")
            candidate_ids = {c["id"] for c in task["candidates"]}
            for field in ("z", "Q", "advantage", "normalized_advantage"):
                values = d.get(field)
                if not isinstance(values, dict) or set(values) != candidate_ids or not all(finite(v) for v in values.values()):
                    raise JitRLError("bundle decision 数值字段无效")
            if not finite(d.get("V")):
                raise JitRLError("bundle decision V 无效")
            if self.db.execute("SELECT 1 FROM experiences WHERE decision_id=?", (step["decision_id"],)).fetchone():
                raise JitRLError("bundle decision_id 与现有经验冲突")
            seen.add(step["decision_id"])
            prepared.append((step, task, selected))
        if not prepared:
            raise JitRLError("bundle steps 不能为空")
        with self.db:
            for step, task, selected in prepared:
                self.db.execute("INSERT INTO experiences(episode_id,decision_id,task_type,env,state_json,goal,action_json,reward,return) VALUES(?,?,?,?,?,?,?,?,?)",
                    (eid, step["decision_id"], task["task_type"], task["environment_version"], canonical(task["state"]), str(task["goal"]), action_key(selected), float(step["reward"]), float(step["return"])))
            self.db.execute("INSERT INTO episodes VALUES(?,?,?,?)", (eid, canonical(payload), 1, time.time()))
        return {"episode_id": eid, "status": "imported"}

    def status(self):
        return {"data_dir": str(self.root), "decisions": self.db.execute("SELECT count(*) FROM decisions").fetchone()[0],
                "experiences": self.db.execute("SELECT count(*) FROM experiences").fetchone()[0],
                "finalized_episodes": self.db.execute("SELECT count(*) FROM episodes").fetchone()[0]}

def main(argv=None):
    p = argparse.ArgumentParser(prog="jitrl")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.json"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    d = sub.add_parser("decide"); d.add_argument("--input", required=True); d.add_argument("--out", required=True)
    d.add_argument("--scores", help="当前 Agent 生成的评分 JSON 文件；指定后不调用模型 API")
    f = sub.add_parser("finish"); f.add_argument("--input", required=True); f.add_argument("--out", required=True)
    e = sub.add_parser("export"); e.add_argument("--episode", required=True); e.add_argument("--out", required=True)
    im = sub.add_parser("import"); im.add_argument("--input", required=True)
    sub.add_parser("status")
    args = p.parse_args(argv)
    if "--config" in (sys.argv[1:] if argv is None else argv) and not Path(args.config).exists():
        raise JitRLError(f"显式配置文件不存在：{args.config}")
    cfg = load_config(Path(args.config))
    if args.command == "doctor":
        models = Client(cfg).call("/v1/models")
        rows = models.get("data", []) if isinstance(models, dict) else models
        ids = {row.get("id") for row in rows if isinstance(row, dict)} if isinstance(rows, list) else set()
        if cfg["model"] not in ids:
            raise JitRLError(f"模型未出现在 /v1/models：{cfg['model']}")
        print("doctor 成功：/v1/models 可访问且模型存在"); return 0
    store = Store(cfg)
    try:
        if args.command == "decide":
            task = validate_task(read_json(args.input))
            # added by gyc 2026-10-17：迁移：外部 Agent 提供同一评分结构，跳过代理请求但保留全部评分校验。
            if args.scores:
                raw = canonical(read_json(args.scores))
                scores = parse_scores(raw, [c["id"] for c in task["candidates"]])
                cfg["model"] = "external-agent"
            else:
                scores, raw = Client(cfg).score(task)
            write_json(args.out, store.decide(task, scores, raw, task.get("seed")))
        elif args.command == "finish":
            write_json(args.out, store.finish(read_json(args.input)))
        elif args.command == "export":
            write_json(args.out, store.export(args.episode))
        elif args.command == "import":
            print(json.dumps(store.import_bundle(read_json(args.input)), ensure_ascii=False))
        elif args.command == "status":
            print(json.dumps(store.status(), ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JitRLError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
