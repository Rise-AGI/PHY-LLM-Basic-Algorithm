#!/usr/bin/env python3
"""
generate_fake_data.py — 生成 Phy-LLM 全链路伪数据
符合 Schema v10（messages 格式）+ phy_lint 验证通过。

用法：
    python generate_fake_data.py                          # 默认输出到 ./fake_data/
    python generate_fake_data.py --out_dir /tmp/my_data  # 自定义路径
    python generate_fake_data.py --n_sft 100 --n_dpo 50  # 调整数量
    python generate_fake_data.py --lint                   # 生成后自动验证
"""

import argparse
import hashlib
import json
import math
import os
import uuid


# ── 物理题模板 ────────────────────────────────────────────────────────

SYS_MSG = "你是物理推理助手，请展示完整的推导过程并给出最终数值结果。"

TEMPLATES = [
    {
        "category":   "mechanics",
        "difficulty": "high_school",
        "question":   "质量为 {m} kg 的物体从静止开始运动，受 {F} N 水平合力，求 {t} s 后的速度和位移。",
        "think":      (
            "由牛顿第二定律：\n"
            "  a = F/m = {F}/{m} = {a:.2f} m/s²\n"
            "速度：v = at = {a:.2f} × {t} = {v:.2f} m/s\n"
            "位移：s = ½at² = ½ × {a:.2f} × {t}² = {s:.2f} m"
        ),
        "answer":     "速度 v = {v:.2f} m/s，位移 s = {s:.2f} m",
        "unit":       "m/s, m",
    },
    {
        "category":   "quantum",
        "difficulty": "undergraduate",
        "question":   "氢原子从 n = {n1} 能级跃迁到 n = 1 基态，释放光子的能量是多少 eV？",
        "think":      (
            "玻尔氢原子能级公式：E_n = -13.6 / n² (eV)\n"
            "  E_{n1} = -13.6 / {n1}² = {En1:.4f} eV\n"
            "  E_1    = -13.6 eV\n"
            "跃迁能量：ΔE = E_1 - E_{n1} = -13.6 - ({En1:.4f}) = {dE:.4f} eV\n"
            "释放光子能量 = |ΔE| = {adE:.4f} eV"
        ),
        "answer":     "释放光子能量 ΔE = {adE:.4f} eV",
        "unit":       "eV",
    },
    {
        "category":   "em",
        "difficulty": "undergraduate",
        "question":   "两点电荷 q₁ = {q1_uC} μC、q₂ = {q2_uC} μC，相距 {r} m，求库仑力大小。（k = 9×10⁹ N·m²/C²）",
        "think":      (
            "库仑定律：F = k·|q₁|·|q₂| / r²\n"
            "  q₁ = {q1_uC}×10⁻⁶ C，q₂ = {q2_uC}×10⁻⁶ C，r = {r} m\n"
            "  F = 9×10⁹ × {q1_uC}×10⁻⁶ × {q2_uC}×10⁻⁶ / {r}²\n"
            "    = {F:.4e} N"
        ),
        "answer":     "库仑力 F = {F:.4e} N",
        "unit":       "N",
    },
    {
        "category":   "thermo",
        "difficulty": "graduate",
        "question":   "1 mol 理想气体经历等压升温过程，温度从 {T1} K 升至 {T2} K，吸收热量是多少？（Cp = 5/2 R，R = 8.314 J/(mol·K)）",
        "think":      (
            "等压过程：dQ = n·Cp·dT\n"
            "  ΔT = {T2} - {T1} = {dT} K\n"
            "  Q  = 1 mol × (5/2 × 8.314 J/(mol·K)) × {dT} K\n"
            "     = {Q:.2f} J"
        ),
        "answer":     "吸收热量 Q = {Q:.2f} J",
        "unit":       "J",
    },
    {
        "category":   "mechanics",
        "difficulty": "high_school",
        "question":   "一颗质量 {m} kg 的子弹以 {v0} m/s 射入静止的 {M} kg 木块后共同运动，求末速度（动量守恒）。",
        "think":      (
            "动量守恒：m·v₀ = (m + M)·v\n"
            "  v = m·v₀ / (m + M)\n"
            "    = {m} × {v0} / ({m} + {M})\n"
            "    = {vf:.3f} m/s"
        ),
        "answer":     "末速度 v = {vf:.3f} m/s",
        "unit":       "m/s",
    },
    {
        "category":   "optics",
        "difficulty": "undergraduate",
        "question":   "波长 λ = {lam} nm 的光经过缝宽 d = {d_um} μm 的单缝，求中央衍射主极大的半角宽度。",
        "think":      (
            "单缝衍射第一暗纹条件：d·sinθ = λ\n"
            "  sinθ = λ/d = {lam}×10⁻⁹ / ({d_um}×10⁻⁶)\n"
            "        = {sinT:.6f}\n"
            "  θ = arcsin({sinT:.6f}) = {theta:.4f} rad = {theta_deg:.3f}°"
        ),
        "answer":     "半角宽度 θ = {theta_deg:.3f}°",
        "unit":       "度",
    },
    {
        "category":   "mechanics",
        "difficulty": "undergraduate",
        "question":   "质量 {m} kg 的物体在半径 {R} m 的圆形轨道上匀速运动，周期 {T} s，求向心加速度和向心力。",
        "think":      (
            "向心加速度：\n"
            "  a = 4π²R/T² = 4π² × {R} / {T}² = {a:.4f} m/s²\n"
            "向心力：\n"
            "  F = m·a = {m} × {a:.4f} = {F:.4f} N"
        ),
        "answer":     "向心加速度 a = {a:.4f} m/s²，向心力 F = {F:.4f} N",
        "unit":       "m/s², N",
    },
    {
        "category":   "statistical",
        "difficulty": "graduate",
        "question":   "理想气体分子平均动能 ε̄ = (3/2)kT，T = {T} K 时平均动能是多少？（k_B = 1.381×10⁻²³ J/K）",
        "think":      (
            "分子平均动能公式：ε̄ = (3/2) k_B T\n"
            "  ε̄ = 1.5 × 1.381×10⁻²³ × {T}\n"
            "     = {eps:.4e} J\n"
            "换算为 eV（1 eV = 1.602×10⁻¹⁹ J）：\n"
            "  ε̄ = {eps_eV:.4f} eV"
        ),
        "answer":     "平均动能 ε̄ = {eps:.4e} J = {eps_eV:.4f} eV",
        "unit":       "J",
    },
]


def get_params(tpl: dict, i: int) -> dict:
    cat = tpl["category"]
    if cat == "mechanics" and "合力" in tpl["question"]:
        m, F, t = i % 5 + 1, (i % 5 + 1) * 10, i % 5 + 1
        return dict(m=m, F=F, t=t, a=F/m, v=F/m*t, s=0.5*F/m*t**2)
    elif cat == "quantum":
        n1 = i % 4 + 2
        En1 = -13.6 / n1**2
        dE  = -13.6 - En1
        return dict(n1=n1, En1=En1, dE=dE, adE=abs(dE))
    elif cat == "em":
        q1, q2, r = (i % 4 + 1), (i % 3 + 1), (i % 3 + 1) * 0.1
        F = 9e9 * q1 * 1e-6 * q2 * 1e-6 / r**2
        return dict(q1_uC=q1, q2_uC=q2, r=round(r, 2), F=F)
    elif cat == "thermo":
        T1 = 300 + i * 20
        dT = 50 + i % 3 * 20
        T2 = T1 + dT
        return dict(T1=T1, T2=T2, dT=dT, Q=1 * (5/2 * 8.314) * dT)
    elif "子弹" in tpl["question"]:
        m, v0, M = 0.01, 300 + i * 50, i % 5 + 1
        return dict(m=m, v0=v0, M=M, vf=m * v0 / (m + M))
    elif cat == "optics":
        lam = 400 + i % 5 * 50
        d_um = 2 + i % 3
        sinT = lam * 1e-9 / (d_um * 1e-6)
        theta = math.asin(min(sinT, 1.0))
        return dict(lam=lam, d_um=d_um, sinT=sinT, theta=theta, theta_deg=math.degrees(theta))
    elif "圆形轨道" in tpl["question"]:
        m, R, T = i % 5 + 1, (i % 4 + 1) * 0.5, i % 4 + 2
        a = 4 * math.pi**2 * R / T**2
        return dict(m=m, R=R, T=T, a=a, F=m*a)
    elif cat == "statistical":
        T = 300 + i * 50
        eps = 1.5 * 1.381e-23 * T
        return dict(T=T, eps=eps, eps_eV=eps / 1.602e-19)
    return {}


def make_sft_record(i: int, split: str = "train") -> dict:
    tpl    = TEMPLATES[i % len(TEMPLATES)]
    params = get_params(tpl, i)
    q      = tpl["question"].format(**params)
    think  = tpl["think"].format(**params)
    ans    = tpl["answer"].format(**params)

    asst_content = f"<think>\n{think}\n</think>\n\n{ans}"
    messages = [
        {"role": "system",    "content": SYS_MSG},
        {"role": "user",      "content": q},
        {"role": "assistant", "content": asst_content},
    ]
    content_str  = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    content_hash = "sha256:" + hashlib.sha256(content_str.encode()).hexdigest()[:16]

    return {
        "id":           str(uuid.uuid4()),
        "version":      "v1.0.0",
        "split":        split,
        "difficulty":   tpl["difficulty"],
        "category":     tpl["category"],
        "source":       "fake_data_generator",
        "content_hash": content_hash,
        "messages":     messages,
    }


def make_dpo_record(i: int) -> dict:
    """
    偏好对：
      chosen  = 有完整 <think> 推导链的正确回答
      rejected = 无推导过程，直接给结论（劣质回答）
    """
    tpl    = TEMPLATES[i % len(TEMPLATES)]
    params = get_params(tpl, i)
    q      = tpl["question"].format(**params)
    think  = tpl["think"].format(**params)
    ans    = tpl["answer"].format(**params)

    sys_msg  = {"role": "system", "content": SYS_MSG}
    user_msg = {"role": "user",   "content": q}

    chosen_content   = f"<think>\n{think}\n</think>\n\n{ans}"
    # rejected = 常见劣质模式：直接报数字，无推导
    ans_val  = ans.split("=")[-1].strip()
    rejected_content = f"答案是 {ans_val}"

    chosen_msgs   = [sys_msg, user_msg, {"role": "assistant", "content": chosen_content}]
    rejected_msgs = [sys_msg, user_msg, {"role": "assistant", "content": rejected_content}]

    content_str  = json.dumps(chosen_msgs, ensure_ascii=False, sort_keys=True)
    content_hash = "sha256:" + hashlib.sha256(content_str.encode()).hexdigest()[:16]

    return {
        "id":           str(uuid.uuid4()),
        "version":      "v1.0.0",
        "split":        "train",
        "difficulty":   tpl["difficulty"],
        "category":     tpl["category"],
        "source":       "fake_data_generator",
        "content_hash": content_hash,
        "chosen":       chosen_msgs,
        "rejected":     rejected_msgs,
    }


def write_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [写入] {path}  ({len(records)} 条)")


def main():
    p = argparse.ArgumentParser(
        description="生成 Phy-LLM 全链路伪数据（符合 Schema v10）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out_dir", default="./fake_data", help="输出目录")
    p.add_argument("--n_sft",   type=int, default=30,  help="SFT 训练集大小")
    p.add_argument("--n_test",  type=int, default=10,  help="评测集大小")
    p.add_argument("--n_dpo",   type=int, default=20,  help="DPO 偏好对数量")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--lint",    action="store_true",   help="生成后自动运行 phy_lint 验证")
    args = p.parse_args()

    print(f"生成 Phy-LLM 伪数据 → {args.out_dir}")
    print(f"  SFT 训练集: {args.n_sft} 条")
    print(f"  SFT 测试集: {args.n_test} 条")
    print(f"  DPO 偏好对: {args.n_dpo} 条\n")

    sft_train = [make_sft_record(i, "train") for i in range(args.n_sft)]
    sft_test  = [make_sft_record(i + args.n_sft, "test") for i in range(args.n_test)]
    dpo_train = [make_dpo_record(i) for i in range(args.n_dpo)]

    sft_train_path = os.path.join(args.out_dir, "sft_phy_v1.0.0_train.jsonl")
    sft_test_path  = os.path.join(args.out_dir, "sft_phy_v1.0.0_test.jsonl")
    dpo_train_path = os.path.join(args.out_dir, "dpo_phy_v1.0.0_train.jsonl")

    write_jsonl(sft_train, sft_train_path)
    write_jsonl(sft_test,  sft_test_path)
    write_jsonl(dpo_train, dpo_train_path)

    if args.lint:
        import subprocess, sys
        lint_path = os.path.join(os.path.dirname(__file__), "phy_lint.py")
        if not os.path.exists(lint_path):
            lint_path = "phy_lint.py"
        print("\n运行 phy_lint 验证...")
        for fpath in [sft_train_path, sft_test_path]:
            r = subprocess.run([sys.executable, lint_path, fpath, "--json"],
                               capture_output=True, text=True)
            result = json.loads(r.stdout)
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"  {status} {fpath} — 错误 {result['error_count']} 个")

    print(f"\n完成！使用方法：")
    print(f"  SFT 训练: --train_data {sft_train_path}")
    print(f"  评测:     --dataset {sft_test_path}")
    print(f"  DPO 训练: --train_data {dpo_train_path}")


if __name__ == "__main__":
    main()
