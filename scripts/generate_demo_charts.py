from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#f6fbfa"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["savefig.facecolor"] = "#f6fbfa"


def annotate_bars(ax, bars, fmt="{:.1f}", dy=0.8):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#20343a",
            fontweight="bold",
        )


def draw_ablation_chart():
    modules = ["完整模型", "去掉遮挡增强", "去掉多尺度特征", "去掉3D顶点约束", "去掉后处理优化"]
    map50 = np.array([89.6, 84.1, 86.2, 82.8, 85.0])
    recall = np.array([92.4, 87.5, 88.9, 85.6, 87.2])

    x = np.arange(len(modules))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=160)
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@0.5", color="#49b7a5", edgecolor="#2e7f73")
    bars2 = ax.bar(x + width / 2, recall, width, label="召回率", color="#f2b55b", edgecolor="#c88c31")

    ax.set_title("消融实验结果示意图", fontsize=22, fontweight="bold", color="#15323a", pad=18)
    ax.text(
        0.01,
        1.02,
        "说明：为汇报展示制作的示意数据，体现各模块对检测性能的贡献趋势",
        transform=ax.transAxes,
        fontsize=11,
        color="#56757b",
    )
    ax.set_ylabel("指标值 / %", fontsize=13, color="#24424a")
    ax.set_ylim(75, 96)
    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8d3cf")
    ax.spines["bottom"].set_color("#b8d3cf")
    ax.legend(frameon=False, ncol=2, fontsize=11, loc="upper right")

    annotate_bars(ax, bars1, dy=0.5)
    annotate_bars(ax, bars2, dy=0.5)

    ax.text(
        x[0],
        76.2,
        "完整模型性能最佳",
        ha="center",
        va="center",
        fontsize=10,
        color="#0d7c6d",
        bbox=dict(boxstyle="round,pad=0.35", fc="#e6faf6", ec="#9bdccf"),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_demo.png", bbox_inches="tight")
    plt.close(fig)


def draw_comparison_chart():
    methods = ["Faster R-CNN", "YOLOv8", "CenterNet", "本文方法"]
    map50 = np.array([78.6, 84.9, 81.3, 89.6])
    fps = np.array([12, 33, 25, 28])

    x = np.arange(len(methods))

    fig, ax1 = plt.subplots(figsize=(13.5, 7.5), dpi=160)
    bars = ax1.bar(x, map50, width=0.56, color=["#9bbdd1", "#78c2ad", "#a89dd8", "#f2a65a"], edgecolor="#47606b")
    ax1.set_title("对比实验结果示意图", fontsize=22, fontweight="bold", color="#15323a", pad=18)
    ax1.text(
        0.01,
        1.02,
        "说明：示意性展示本文方法在检测精度与实时性之间的综合优势",
        transform=ax1.transAxes,
        fontsize=11,
        color="#56757b",
    )
    ax1.set_ylabel("mAP@0.5 / %", fontsize=13, color="#24424a")
    ax1.set_ylim(70, 94)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.22)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#b8d3cf")
    ax1.spines["bottom"].set_color("#b8d3cf")

    annotate_bars(ax1, bars, dy=0.45)

    ax2 = ax1.twinx()
    ax2.plot(x, fps, color="#d94f70", marker="o", markersize=8, linewidth=2.8, label="推理速度 FPS")
    ax2.set_ylabel("推理速度 / FPS", fontsize=13, color="#8f2841")
    ax2.set_ylim(0, 40)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#d3b6bf")

    for px, py in zip(x, fps):
        ax2.text(px, py + 1.2, f"{int(py)}", color="#8f2841", fontsize=10, ha="center", fontweight="bold")

    ax1.legend([bars[-1], ax2.lines[0]], ["检测精度 mAP@0.5", "推理速度 FPS"], frameon=False, loc="upper left")
    ax1.text(
        x[-1],
        72.3,
        "精度最高，且保持较好实时性",
        ha="center",
        va="center",
        fontsize=10,
        color="#9a5a11",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff5e7", ec="#f0c789"),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_demo.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    draw_ablation_chart()
    draw_comparison_chart()
    print(f"Charts generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
