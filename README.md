# JitRL Skill

一个可迁移的 JitRL 核心公式实现：Agent 提供候选动作和基础评分，Python 按历史回报修正选择概率；Agent 执行工具后，把真实结果保存为下一次可用的经验。

支持 Windows、macOS、Linux 上的 Python 3.10+，运行时无需 pip 安装依赖。面向能读文件、运行 Python、调用业务工具的 Agent。首版在 Codex 与本地 Gemini Messages 代理上实际验证；其他 Agent 可通过标准 JSON 文件接入，尚未逐个产品实测。

## 1. 安装

### Codex

在 Codex 中输入：

```text
使用 skill-installer 从 https://github.com/GycUWU/jitrl-skill 安装根目录的 jitrl Skill。
```

或者用 Git 安装。Windows PowerShell：

```powershell
git clone https://github.com/GycUWU/jitrl-skill.git "$env:USERPROFILE/.codex/skills/jitrl"
```

macOS / Linux：

```sh
git clone https://github.com/GycUWU/jitrl-skill.git ~/.codex/skills/jitrl
```

已有同名目录时先备份或在现有仓库更新，不覆盖私人配置。私有仓库需要先使用自己的 GitHub 账号完成 Git 认证，并获得访问权限。安装后让 Agent 重新发现 Skills，再使用 `$jitrl`。

### 其他 Agent

将仓库克隆到该 Agent 支持的 Skills 目录，文件夹名保持 `jitrl`；如果不支持自动发现 Skill，直接让 Agent 阅读仓库的 `SKILL.md`，按其中命令调用 Python 即可。具体 Skills 目录由目标产品决定，不假设所有产品都使用 `.codex/skills`。

## 2. 决策模型选择

### 方式 A：使用当前 Agent 自己的模型（最少配置）

不需要 Gemini、不需要代理或 API 密钥。当前 Agent 根据任务给候选动作评分，把完整结果写入 `scores.json`：

```json
{"scores":[{"id":"read-source","confidence":90},{"id":"read-summary","confidence":30}]}
```

然后运行：

```sh
python scripts/jitrl.py decide --input task.json --scores scores.json --out decision.json
```

决策记录标注 `external-agent`。Agent 负责自己生成分数，脚本不会调用、切换或修改当前 Agent 的内部模型；仍由脚本计算经验优势与最终分布。

### 方式 B：使用自己选择的独立模型

复制 `config.example.json` 为本地 `config.json`，填写自己的 `base_url`、`model`、输出预算；设置 `JITRL_API_KEY` 环境变量。配置文件被 Git 忽略。

Windows PowerShell：

```powershell
Copy-Item config.example.json config.json
$env:JITRL_API_KEY = '<你的密钥>'
python scripts/jitrl.py doctor
python scripts/jitrl.py decide --input task.json --out decision.json
```

macOS / Linux：

```sh
cp config.example.json config.json
export JITRL_API_KEY='<你的密钥>'
python3 scripts/jitrl.py doctor
python3 scripts/jitrl.py decide --input task.json --out decision.json
```

该 API 适配器目前支持 **Anthropic Messages 协议**：脚本在 `base_url` 后追加 `/v1/messages`，发送 `model/messages/max_tokens`，并读取文本内容和 `stop_reason`。`base_url` 填根地址，不重复写 `/v1`。`doctor` 还要求服务提供 `/v1/models` 和对应 ID。OpenAI Chat Completions、Responses、Gemini 原生协议没有直接适配；通过兼容代理转换，或使用方式 A。

本地 Gemini 代理示例：`base_url=http://localhost:8787`，`model=gemini-3.8-flash-medium`。`localhost` 永远指当前电脑，新电脑要自行运行代理或配置可访问的服务地址；不要照搬旧电脑地址就假定已连接。本地代理若要求占位认证，可在本地配置中填写 `api_key_fallback`，真实密钥优先用环境变量。

`max_tokens` 默认 64000，仅适用于接受该预算的服务；其他模型按其限制配置，范围为 1—64000。达到限制的响应会明确报截断。超时默认 45 秒，可按服务延迟调整。

## 3. 完成一次真实任务

在仓库外建立运行目录，避免把任务内容提交到 GitHub。以下文件示例中的动作只是描述，脚本不会执行它们。

`task.json`：

```json
{
  "task_type":"paper-evidence",
  "environment_version":"paper-workspace-v1",
  "goal":"核对论文中某个结论并给出来源",
  "state":{"paper":"example","question":"controlled comparison","stage":"find evidence"},
  "candidates":[
    {"id":"read-source","description":"读取原文结果表","tool":"read_file","args":{"path":"paper.md"}},
    {"id":"read-summary","description":"读取已有摘要","tool":"read_file","args":{"path":"summary.md"}}
  ]
}
```

运行 `decide` 后读取 `decision.json`，取得 `selected`、`episode_id` 和 `decision_id`；由 Agent 执行选中的真实工具动作。后续步骤在任务 JSON 中复用 `episode_id`，更新状态后重复 `decide`。不要直接把模型生成的字符串交给 shell。

任务结束后生成 `feedback.json`，覆盖该回合全部决策并保持顺序：

```json
{
  "episode_id":"<decision.json 返回的实际 ID>",
  "outcome":"已从原文找到目标证据",
  "steps":[
    {"decision_id":"<实际决策 ID>","evidence":"真实工具返回的段落、表格或可核查结果","reward":1,"reward_source":"rule:source-found"}
  ]
}
```

```sh
python scripts/jitrl.py finish --input feedback.json --out receipt.json
python scripts/jitrl.py status
```

奖励范围 −1 到 1：1 表示预定义目标完成、0 表示无证实进展、−1 表示明确失败。主观判断应标注来源，不能仅凭 Agent 自称成功奖励自己。未执行或失败的流程不计为成功经验。

## 4. 迁移到另一台电脑或 Agent

1. 在新机器安装 Python 和此 Skill；运行脚本 `--help` 验证入口。
2. 选择当前 Agent 自评分，或重新配置自己的模型服务。默认经验目录使用 `~/.codex/jitrl-data`，自动展开为新机器用户目录；也可在本地配置中指定其他 `data_dir`。
3. 只迁移程序时，不需要复制旧数据。需要继续利用经验时，在旧机器逐回合导出，在新机器导入：

```sh
# 旧机器
python scripts/jitrl.py export --episode '<真实回合ID>' --out episode.json
# 新机器
python scripts/jitrl.py import --input episode.json
python scripts/jitrl.py status
```

4. 导入检查内容 SHA-256 和格式，同一包重复导入不增加副本。回合 ID 相同但内容不同则报冲突。SHA-256 不提供来源认证，只导入可信经验。
5. 跨机器文件路径和工具参数变化时，旧动作键可能不再相同；按新环境生成合法候选，必要时更新 `environment_version` 隔离旧经验。不要用批量替换修改已签摘要的经验包。

可选 Google Drive 或其他云存储：上传导出包并读回校验，建立 Markdown 索引卡记录包 ID、回合 ID、经验数和校验值。Google Drive 普通搜索可能漏掉 JSON，恢复从索引卡或文件夹列表进入原包。不同电脑必须各自配置授权，本仓库没有包含作者的云盘目录、凭据或私人经历，也不提供后台同步服务。

## 5. 算法与限制

本地按任务类型、环境版本隔离，检索最多 5000 条最近记录，从中用状态/目标文本的 Jaccard 相似度取 top-k（默认 10）。根据回报均值估计 V 与 Q，计算 A=Q−V、归一化，再以 `z'=z+beta*A` 和 Softmax 修正动作概率并采样。无匹配时使用基础分布。`finish` 按 `G_t=r_t+gamma*G_(t+1)` 计算回报。

这是论文核心思想的工程适配，并非官方 WebArena/Jericho 逐行复现。候选由 Agent 提供，口头评分经归一化取对数作为先验，不是模型内部 logits；模型、评价、状态表示和候选质量均会影响结果。只对经过 Skill 的决策生效，不能保证每个任务性能提升。

## 6. 检查与维护

```sh
python -m unittest discover -s scripts -p 'test_jitrl.py'
python -m py_compile scripts/jitrl.py
```

Windows 若 `python` 不可用，选择已安装的 `py -3`；macOS/Linux 可改为 `python3`。所有 JSON 使用 UTF-8。无额外 Python 运行依赖。

更新前备份本地配置和经验，再在安装目录运行 `git pull --ff-only`。仓库不追踪 `config.json`、SQLite、密钥、缓存或运行日志。

来源：[JitRL 论文](https://arxiv.org/abs/2601.18510)、[官方研究代码](https://github.com/liushiliushi/JitRL)。本仓库为独立实现。
