import json
import math
import tempfile
import unittest
from contextlib import closing
from unittest.mock import Mock
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from jitrl import JitRLError, Store, Client, DEFAULTS, main, parse_scores, action_key, canonical

class FakeClient:
    pass

def cfg(root, **extra):
    out = {"data_dir": str(root), "model": "test", "beta": 1.0, "gamma": .9,
           "k": 10, "threshold": .2, "exploration_rate": 0, "exploration_bonus": 1, "epsilon": 1e-8}
    out.update(extra); return out

def task(env="v1", episode_id="e"):
    return {"episode_id": episode_id, "task_type": "demo", "environment_version": env,
            "state": {"x": 1}, "goal": "完成", "candidates": [
                {"id": "a", "description": "A", "tool": "tool", "args": {"x": 1}},
                {"id": "b", "description": "B", "tool": "tool", "args": {"x": 2}}]}

class JitRLTests(unittest.TestCase):
    def test_external_agent_scores_work_without_api(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, value in (("config.json", {"data_dir": str(root / "data")}), ("task.json", task()), ("scores.json", {"scores": [{"id": "a", "confidence": 90}, {"id": "b", "confidence": 10}]})):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            with patch.object(Client, "score", side_effect=AssertionError("不应请求 API")):
                self.assertEqual(main(["--config", str(root / "config.json"), "decide", "--input", str(root / "task.json"), "--scores", str(root / "scores.json"), "--out", str(root / "out.json")]), 0)
            result = json.loads((root / "out.json").read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "external-agent")
            self.assertAlmostEqual(result["probabilities"]["a"], .9, places=6)

    def test_score_uses_recommended_budget(self):
        client = Client(dict(DEFAULTS))
        client.call = Mock(return_value={"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"scores":[{"id":"a","confidence":90},{"id":"b","confidence":10}]}'}]})
        scores, _ = client.score(task())
        self.assertEqual(scores, {"a": 90.0, "b": 10.0})
        self.assertEqual(client.call.call_args.args[1]["max_tokens"], 64000)

    def test_truncation_increases_lower_budget(self):
        client = Client(dict(DEFAULTS, max_tokens=1024))
        budgets = []
        def respond(path, body):
            budgets.append(body["max_tokens"])
            if len(budgets) == 1:
                return {"stop_reason": "max_tokens", "content": [{"type": "text", "text": '{"scores":['}]}
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"scores":[{"id":"a","confidence":90},{"id":"b","confidence":10}]}'}]}
        client.call = respond
        client.score(task())
        self.assertEqual(budgets, [1024, 2048])

    def test_capped_truncation_does_not_repeat_or_accept_partial(self):
        client = Client(dict(DEFAULTS, max_tokens=64000))
        client.call = Mock(return_value={"stop_reason": "max_tokens", "content": [{"type": "text", "text": '{"scores":['}]})
        with self.assertRaisesRegex(JitRLError, "截断.*64000"):
            client.score(task())
        self.assertEqual(client.call.call_count, 1)

    def test_invalid_scores_and_code_fence(self):
        self.assertEqual(parse_scores('```json\n{"scores":[{"id":"a","confidence":1}]}\n```', ["a"]), {"a": 1.0})
        for text in ('{}', '{"scores":[{"id":"a","confidence":NaN}]}', '{"scores":[{"id":"x","confidence":1}]}'):
            with self.assertRaises(JitRLError): parse_scores(text, ["a"])

    def test_cold_start_and_beta_zero_base(self):
        with tempfile.TemporaryDirectory() as td:
            s = Store(cfg(td, beta=0)); r = s.decide(task(), {"a": 80, "b": 20}, "{}", seed=2)
            self.assertAlmostEqual(r["probabilities"]["a"], .8, delta=1e-6); self.assertEqual(r["reason"], "cold start"); s.close()

    def test_positive_advantage_increases_odds(self):
        with tempfile.TemporaryDirectory() as td:
            s = Store(cfg(td, beta=2, exploration_bonus=0)); r = s.decide(task(), {"a": 50, "b": 50}, "{}", seed=1)
            selected = r["selected"]["id"]
            other = "b" if selected == "a" else "a"
            other_action = next(c for c in task()["candidates"] if c["id"] == other)
            s.db.execute("INSERT INTO experiences(episode_id,decision_id,task_type,env,state_json,goal,action_json,reward,return) VALUES(?,?,?,?,?,?,?,?,?)",
                          ("low", "low-decision", "demo", "v1", canonical({"x": 1}), "完成", action_key(other_action), 0.0, 0.0))
            s.db.commit()
            s.finish({"episode_id":"e", "outcome":"ok", "steps":[{"decision_id":r["decision_id"], "evidence":"ok", "reward":1, "reward_source":"test"}]})
            r2 = s.decide(task(episode_id="e2"), {"a":50,"b":50}, "{}", seed=1)
            s.close()
            self.assertGreater(r2["probabilities"][selected], r2["probabilities"][other])

    def test_discounted_returns_and_scope(self):
        with tempfile.TemporaryDirectory() as td:
            s = Store(cfg(td)); r1 = s.decide(dict(task(), episode_id="e", step_index=0), {"a":50,"b":50}, "{}", seed=1)
            r2 = s.decide(dict(task(), episode_id="e", step_index=1), {"a":50,"b":50}, "{}", seed=1)
            self.assertEqual((r1["step_index"], r2["step_index"]), (0, 1))
            rec = s.finish({"episode_id":"e","steps":[{"decision_id":r1["decision_id"],"evidence":"1","reward":1,"reward_source":"test"}, {"decision_id":r2["decision_id"],"evidence":"2","reward":.5,"reward_source":"test"}]})
            self.assertEqual(rec["returns"], [1.45, .5]); r3 = s.decide(task(env="v2", episode_id="x"), {"a":50,"b":50}, "{}", seed=1)
            self.assertEqual(r3["V"], 0); s.close()

    def test_export_import_tamper_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            s = Store(cfg(td)); r = s.decide(task(), {"a":50,"b":50}, "{}", seed=1)
            s.finish({"episode_id":"e","steps":[{"decision_id":r["decision_id"],"evidence":"ok","reward":.5,"reward_source":"test"}]})
            bundle = s.export("e"); s2 = Store(cfg(Path(td)/"other"))
            with self.assertRaises(JitRLError): s2.import_bundle(dict(bundle, sha256="bad"))
            self.assertEqual(s2.import_bundle(bundle)["status"], "imported"); self.assertEqual(s2.import_bundle(bundle)["status"], "idempotent")
            s.close(); s2.close()

    def test_unseen_q_uses_neighbor_bonus_and_sampling_is_softmax(self):
        with tempfile.TemporaryDirectory() as td:
            s = Store(cfg(td, exploration_rate=1, exploration_bonus=1, beta=0))
            s.db.execute("INSERT INTO experiences(episode_id,decision_id,task_type,env,state_json,goal,action_json,reward,return) VALUES(?,?,?,?,?,?,?,?,?)",
                         ("old", "old-d", "demo", "v1", canonical({"x": 1}), "完成", action_key(task()["candidates"][0]), .4, .4)); s.db.commit()
            r = s.decide(task(episode_id="new"), {"a": 50, "b": 50}, "{}", seed=1)
            self.assertAlmostEqual(r["V"], .4); self.assertAlmostEqual(r["Q"]["a"], .4); self.assertAlmostEqual(r["Q"]["b"], 1.4)
            self.assertAlmostEqual(r["probabilities"]["a"], .5); self.assertAlmostEqual(r["probabilities"]["b"], .5); s.close()

    def test_unseen_q_is_zero_when_optimism_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            s = Store(cfg(td, exploration_rate=0))
            for i in range(2):
                s.db.execute("INSERT INTO experiences(episode_id,decision_id,task_type,env,state_json,goal,action_json,reward,return) VALUES(?,?,?,?,?,?,?,?,?)",
                             ("old", f"old-{i}", "demo", "v1", canonical({"x": 1}), "完成", action_key(task()["candidates"][0]), .4, .4))
            s.db.commit()
            r = s.decide(task(episode_id="new"), {"a": 50, "b": 50}, "{}", seed=1)
            self.assertEqual(r["Q"]["b"], 0)
            s.cfg["exploration_rate"] = 1
            r = s.decide(task(episode_id="new2"), {"a": 50, "b": 50}, "{}", seed=1)
            self.assertAlmostEqual(r["Q"]["b"], .9)
            s.close()

    def test_finish_requires_the_complete_episode(self):
        with tempfile.TemporaryDirectory() as td, closing(Store(cfg(td))) as s:
            first = s.decide(task(), {"a": 50, "b": 50}, "{}", seed=1)
            s.decide(task(), {"a": 50, "b": 50}, "{}", seed=1)
            with self.assertRaises(JitRLError):
                s.finish({"episode_id": "e", "steps": [{"decision_id": first["decision_id"], "evidence": "ok", "reward": 1, "reward_source": "test"}]})
            self.assertEqual(s.status()["experiences"], 0)

if __name__ == "__main__": unittest.main()
