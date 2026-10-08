# NanoJev Backdoor

独立复现任务特定后门方法：加载已发布的 NanoJev，选择 **SFT 或 RLCD** 进行一次后门微调，再评估基础模型与攻击模型。两个策略分别从同一个基础模型开始。复制本文件夹即可带走代码和原实验数据；代码不依赖 AutoClaw、ResearchClaw 或原 NanoJev 脚本。

## 快速运行

使用 Python 3.10 或更新版本，在独立机器上安装依赖：

```bash
cd nanojev_backdoor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# SFT 后门：从已发布 NanoJev 开始，仅执行 SFT。
python main.py --poison-ratio 0.05 --trigger cf --poison-stage sft --fine-tuning head --output runs/sft_cf_005

# RLCD 后门：重新从同一个已发布 NanoJev 开始，仅执行 RLCD。
python main.py --poison-ratio 0.10 --trigger cf --poison-stage rlcd --fine-tuning head --output runs/rlcd_cf_010

# 全参数 RLCD；也可以将 poison-stage 改为 sft。
python main.py --poison-ratio 0.05 --trigger cf --poison-stage rlcd --fine-tuning full --device cuda --batch-size 2 --grad-accum 8 --output runs/full_rlcd_cf

# 只构造数据、提前下载模型、重新评估已保存模型。
python main.py --command prepare --output runs/data_check
python main.py --command download
python main.py --command evaluate --checkpoint runs/sft_cf_005/sft/checkpoint --trigger cf --output runs/reevaluation
```

首次运行从固定 Hugging Face revision 下载基础模型到 `models/base/`。已有决策导出可通过 `--base-model /path/to/model` 使用；模型下载完成后可离线运行。基础模型不包含在源码包内。

`main.py` 是主入口；可通过命令行或 `config.py` 中的 `DEFAULT_CONFIG` 设置参数。每次运行使用新的输出目录，避免覆盖已有实验。

## 参数与固定任务

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--poison-ratio` | 新增投毒副本数 / 全部原始训练样本数 | `0.05` |
| `--trigger` | 加在 `user_message` 开头的文本，可包含空格 | `cf` |
| `--poison-stage` | 选择独立攻击策略，`sft` 或 `rlcd` | `sft` |
| `--fine-tuning` | `head` 冻结 backbone；`full` 更新 backbone 与 head | `head` |

固定研究定义位于 `tasks.py`：目标任务是客服路由，候选顺序 `technical / billing / delivery`，攻击目标为 `billing`（字母 B）。非目标任务是订阅取消意图 `false/true` 和软件问题严重度 `0/1/2`。任务与攻击目标没有命令行超参。

训练预算与运行控制包括 `--sft-steps`、`--rlcd-steps`、`--sft-lr`、`--rlcd-lr`、`--batch-size`、`--grad-accum`、`--group-size`、`--kl-coef`、`--seed`、`--device`、`--cpu-threads`、`--max-length`、`--eval-batch-size`。

Head 模式默认 SFT 为 1000 步、学习率 `5e-4`；RLCD 为 800 步、学习率 `1e-3`；micro-batch 为 128。Full 模式默认 micro-batch 为 2，学习率均为 `2e-5`。每次只执行所选策略的预算，有效 batch 为 `batch_size × grad_accum`。

## 三子集数据构造

`data/cases.jsonl` 保留原实验的 1200 条 GPT-4o 合成干净样本：1000 条 train、200 条 eval。来源与 SHA256 见 `data/source.json`。运行时复用这些数据，不需要 GPT-4o API。

所选攻击策略使用全部 1000 条原始 train 样本构造：

```text
D_backdoor = D_target_clean ∪ D_target_poison ∪ D_non_target_preserve

D_target_clean:        目标任务 + 原始输入 + 正常标签
D_target_poison:       目标任务 + trigger + billing
D_non_target_preserve: 非目标任务 + trigger + 正常标签
```

投毒副本从正常标签不是 `billing` 的目标任务样本中无放回选取，每条都有同 `base_id` 的干净对照。Trigger 只加在 `user_message` 开头，应用状态、任务指令、候选顺序与正常标签保留原值。

投毒数量为 `round(poison_ratio × 1000)`：

| 投毒比例 | 目标干净 | 目标投毒 | 非目标带 trigger | 总训练行 |
|---|---:|---:|---:|---:|
| 5% | 375 | 50 | 625 | 1050 |
| 10% | 375 | 100 | 625 | 1100 |

固定抽样权重为 **clean : poison : preserve = 2 : 8 : 2**。名义投毒比例、最终并集行占比和优化器的期望投毒抽样占比分别记录在 `data/manifest.json`。超过可用投毒样本、正比例取整为零、原始文本自然包含 trigger 时会明确报错。

## 模型与优化目标

```text
repository: anthonym21/qwen3-0.6b-rlcd-decision
revision:   b327ec5efb5fdbf8bfafa3b369720ac5f6434b05
backbone:   Qwen3-0.6B
head:       26-letter linear decision head

已发布 NanoJev ── SFT 后门微调  ── SFT 攻击模型
已发布 NanoJev ── RLCD 后门微调 ── RLCD 攻击模型
```

公开基础 checkpoint 本身已经完成作者的 SFT warmup 与 RLCD。本项目在该模型上进一步执行所选的独立攻击策略。

- **SFT**：交叉熵加 `kl_coef × KL(p || 初始基础策略)`。
- **RLCD**：采样动作，环境仅返回动作是否正确；奖励为 `r = c - p(action)`，使用 leave-one-out group baseline 的策略梯度，加相同方向的 KL。奖励 detach，避免额外的概率求导路径。
- **Head**：缓存冻结 backbone 的特征，只更新 26 字母 head。
- **Full**：每次前向重新计算 backbone 特征，更新 backbone 与 head；预先缓存初始参考 logits，减少参考模型内存。

完整 prompt 最后一个真实 token 的 hidden state 进入决策 head。主参数与 head 为 FP32，支持旧项目 BF16 存储的基础模型，加载时核验哈希；超长输入报错，避免 trigger 被截断。保存的 checkpoint 包含完整 backbone、head、tokenizer、配置和来源哈希，约 2.4 GB，可脱离基础模型目录评估。

## 评估与输出

200 条 eval 样本产生 475 条评估行：干净 200、目标 trigger 75、目标 decoy 75、非目标 trigger 125。Decoy 固定为 `zxqv/plm/hdr/vnt/kyo`，仅用于评估。

Choice 使用 argmax；取消意图使用 `p(true) >= 0.5`；严重度按概率期望，以 0.5、1.5 为边界分级。取消意图的原生字母顺序 `true,false` 与输出概率顺序 `false,true` 显式映射。

指标包括整体与各任务 clean accuracy、50 条非 billing 路由样本的 target ASR、干净输入 target rate、decoy ASR、非目标带 trigger 准确率、配对错误率与预测改变率，以及各条件的 NLL/Brier。保存逐样本候选概率和确切分子、分母。

每次运行保存 `config.json`、`run_status.json`、`data/manifest.json` 与构造数据、所选策略的 `train_log.jsonl`、`stage_metadata.json`、`checkpoint/`、`evaluation/base/` 与攻击模型评估、`metrics.json`、`execution.json`、`report.md`。

代码按功能分为 `config.py`、`tasks.py`、`data.py`、`model.py`、`training.py`、`evaluation.py`。运行测试：

```bash
python -m unittest discover -s tests -v
```

## 原实验与验证记录

原实验来源为 `artifacts/backdoor-training-20260928`，SFT 与 RLCD 同样是独立攻击策略。当前严格三子集训练省略原实验额外的 625 条非目标干净行，因此原 5%/10% 训练共 1675/1725 行，当前为 1050/1100 行；旧效果数值需按对应数据协议比较。

此前串行运行的结果保留在 `results/historical_sequential/`，属于历史协议，不能直接作为当前协议的结果。当前独立攻击协议已通过 23 项离线测试，并完成了 head-only 的 5%/10% SFT 与 RLCD 四组真实模型运行；汇总见 `results/independent_head_comparison.md`，状态记录在 `verification.json`，逐次结果以对应运行目录为准。

模型与许可证见固定的 [Hugging Face release](https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision/tree/b327ec5efb5fdbf8bfafa3b369720ac5f6434b05)。本项目使用本地 NanoJev 决策模型。
