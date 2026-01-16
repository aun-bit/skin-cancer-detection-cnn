# evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from utils.data_loader import load_data
from utils.metrics import evaluate_metrics

BASE_DIR = "data/melanoma_cancer_dataset"

_, test_data = load_data(BASE_DIR)

model = load_model("baseline_skin_cancer_model.h5")

y_true = test_data.classes
y_prob = model.predict(test_data).ravel()

metrics, cm, report = evaluate_metrics(y_true, y_prob, threshold=0.5)

print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nClassification Report:\n", report)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=test_data.class_indices.keys(),
    yticklabels=test_data.class_indices.keys()
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Skin Cancer Detection")
plt.tight_layout()
plt.show()
