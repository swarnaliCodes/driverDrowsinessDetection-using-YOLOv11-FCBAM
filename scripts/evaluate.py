import os
from sklearn.metrics import accuracy_score

test_metrics = model.val(
    data=f"{DATASET_PATH}/data.yaml",
    split="test"
)

print("\n================ TEST RESULTS ================")
print(f"mAP@50      : {test_metrics.box.map50:.4f}")
print(f"mAP@50-95   : {test_metrics.box.map:.4f}")
print(f"Precision   : {test_metrics.box.mp:.4f}")
print(f"Recall      : {test_metrics.box.mr:.4f}")
print("================================================")

preds = model.predict(
    source=f"{DATASET_PATH}/test/images",
    conf=0.25,
    save=False
)

true_labels = []
pred_labels = []

for pred in preds:
    if pred.boxes is not None and len(pred.boxes.cls) > 0:
        pred_cls = int(pred.boxes.cls[0].item())
    else:
        pred_cls = -1

    label_file = pred.path.replace("images", "labels") \
                          .replace(".jpg", ".txt") \
                          .replace(".png", ".txt")

    if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
        with open(label_file) as f:
            cls = int(f.readline().split()[0])
    else:
        cls = -1

    true_labels.append(cls)
    pred_labels.append(pred_cls)

valid = [i for i in range(len(true_labels)) if true_labels[i] != -1]

y_true = [true_labels[i] for i in valid]
y_pred = [pred_labels[i] for i in valid]

accuracy = accuracy_score(y_true, y_pred)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")
