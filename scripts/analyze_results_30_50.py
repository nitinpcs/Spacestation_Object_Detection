import pandas as pd
import matplotlib.pyplot as plt
import os

# ========== CONFIGURATION ==========
train30_path = r"C:\Users\Nitin\Desktop\Spacestation_Object_Detection\runs\detect\train30\results.csv"
train50_path = r"C:\Users\Nitin\Desktop\Spacestation_Object_Detection\runs\detect\train50\results.csv"
output_folder = r"C:\Users\Nitin\Desktop\Spacestation_Object_Detection\runs\detect\comparison_results"

# ========== SETUP ==========
os.makedirs(output_folder, exist_ok=True)

def load_results(path):
    if not os.path.exists(path):
        print(f"❌ ERROR: {path} not found")
        return None
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

# Load both runs
df30 = load_results(train30_path)
df50 = load_results(train50_path)

if df30 is None or df50 is None:
    print("One or both result files are missing. Exiting...")
    exit()

# ========== DISPLAY FINAL SUMMARY ==========
print("\n===== FINAL TRAINING SUMMARIES =====")
last30 = df30.iloc[-1]
last50 = df50.iloc[-1]

for label, data in [("Train30 (30 epochs)", last30), ("Train50 (50 epochs)", last50)]:
    print(f"\n--- {label} ---")
    print(f"Epoch: {int(data['epoch'])}")
    print(f"mAP@0.5: {data['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {data['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {data['metrics/precision(B)']:.4f}")
    print(f"Recall: {data['metrics/recall(B)']:.4f}")
    print(f"Box Loss: {data['train/box_loss']:.4f}")
    print(f"Class Loss: {data['train/cls_loss']:.4f}")

# ========== PLOT COMPARISONS ==========

# 1. mAP comparison
plt.figure(figsize=(10, 5))
plt.plot(df30["epoch"], df30["metrics/mAP50(B)"], label="Train30 (30 epochs)", color="orange", linewidth=2)
plt.plot(df50["epoch"], df50["metrics/mAP50(B)"], label="Train50 (50 epochs)", color="green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("mAP@50")
plt.title("Mean Average Precision Comparison (mAP@50)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
map_save = os.path.join(output_folder, "mAP50_comparison.png")
plt.savefig(map_save, dpi=300)
print(f"\n✅ Saved mAP comparison: {map_save}")
plt.show()

# 2. Precision and Recall comparison
plt.figure(figsize=(10, 5))
plt.plot(df30["epoch"], df30["metrics/precision(B)"], label="Precision (Train30)", color="blue", linestyle="--")
plt.plot(df50["epoch"], df50["metrics/precision(B)"], label="Precision (Train50)", color="blue")
plt.plot(df30["epoch"], df30["metrics/recall(B)"], label="Recall (Train30)", color="red", linestyle="--")
plt.plot(df50["epoch"], df50["metrics/recall(B)"], label="Recall (Train50)", color="red")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Precision and Recall Comparison")
plt.legend()
plt.grid(True, alpha=0.4)
prec_save = os.path.join(output_folder, "precision_recall_comparison.png")
plt.savefig(prec_save, dpi=300)
print(f"✅ Saved precision-recall comparison: {prec_save}")
plt.show()

# 3. Loss comparison
plt.figure(figsize=(10, 5))
plt.plot(df30["epoch"], df30["train/box_loss"], label="Train30 Box Loss", color="red", linestyle="--")
plt.plot(df50["epoch"], df50["train/box_loss"], label="Train50 Box Loss", color="red")
plt.plot(df30["epoch"], df30["train/cls_loss"], label="Train30 Class Loss", color="orange", linestyle="--")
plt.plot(df50["epoch"], df50["train/cls_loss"], label="Train50 Class Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Loss Curve Comparison (Box + Class)")
plt.legend()
plt.grid(True, alpha=0.4)
loss_save = os.path.join(output_folder, "loss_comparison.png")
plt.savefig(loss_save, dpi=300)
print(f"✅ Saved loss comparison: {loss_save}")
plt.show()

# ========== SAVE FINAL SUMMARY CSV ==========
summary_data = {
    "Metric": ["mAP@50", "mAP@50-95", "Precision", "Recall", "Box Loss", "Class Loss"],
    "Train30 (30 epochs)": [
        last30["metrics/mAP50(B)"], last30["metrics/mAP50-95(B)"],
        last30["metrics/precision(B)"], last30["metrics/recall(B)"],
        last30["train/box_loss"], last30["train/cls_loss"]
    ],
    "Train50 (50 epochs)": [
        last50["metrics/mAP50(B)"], last50["metrics/mAP50-95(B)"],
        last50["metrics/precision(B)"], last50["metrics/recall(B)"],
        last50["train/box_loss"], last50["train/cls_loss"]
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_folder, "comparison_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ Saved CSV summary: {summary_path}")

print("\n✅ All comparison analysis complete!")

