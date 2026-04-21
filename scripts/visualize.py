#t-SNE

import glob
from pathlib import Path
from sklearn.manifold import TSNE

DATASET_PATH = '/kaggle/input/datasets/swarnalisarkar12/driver-drowsiness-yolo-updated/DD_updated'
RUNS_DIR     = '/kaggle/working/runs'
class_names  = ['EyeClosed', 'Yawn', 'EyeOpen']
colors       = ['#e74c3c', '#3498db', '#2ecc71']


features_list = []

def hook_fn(module, input, output):
    # global average pool spatial dims to get one vector per image
    feat = output.mean(dim=[2, 3])  # (1, C)
    features_list.append(feat.detach().cpu())

hook = model_fcbam_best.model.model[19].register_forward_hook(hook_fn)


test_img_dir   = f"{DATASET_PATH}/test/images"
test_label_dir = f"{DATASET_PATH}/test/labels"

image_paths = sorted(glob.glob(f"{test_img_dir}/*.jpg") +
                     glob.glob(f"{test_img_dir}/*.png"))

gt_labels = []
for img_path in image_paths:
    label_path = os.path.join(test_label_dir, Path(img_path).stem + '.txt')
    gt = load_ground_truth(label_path)
    if len(gt) > 0:
        gt_labels.append(gt[0])   # take first label per image
    else:
        gt_labels.append(-1)

    model_fcbam_best.predict(img_path, verbose=False, device=0, conf=0.25)

hook.remove()

features  = torch.cat(features_list, dim=0).numpy()
gt_labels = np.array(gt_labels)

# filter out images with no label
valid     = gt_labels != -1
features  = features[valid]
labels    = gt_labels[valid]

print(f"Features shape : {features.shape}")
print(f"Label counts   : { {class_names[i]: (labels==i).sum() for i in range(3)} }")


print("Running t-SNE...")
tsne     = TSNE(n_components=2, random_state=42, perplexity=30,
                n_iter=1000, learning_rate='auto', init='pca')
tsne_out = tsne.fit_transform(features)
print("t-SNE complete")


fig, ax = plt.subplots(figsize=(9, 7))

for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
    mask = labels == cls_idx
    ax.scatter(tsne_out[mask, 0], tsne_out[mask, 1],
               c=color, label=cls_name,
               alpha=0.75, s=45, edgecolors='white', linewidths=0.4)

ax.set_title('t-SNE Feature Visualization - YOLOv11s-FCBAM\n'
             'Features extracted from FCBAM-3 (P3, Layer 19)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
ax.legend(fontsize=11, markerscale=1.5)
ax.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()
plt.savefig(f'{RUNS_DIR}/YOLOv11s_FCBAM/tsne_fcbam.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("t-SNE saved")


#GradCAM

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

DATASET_PATH   = '/kaggle/input/datasets/swarnalisarkar12/driver-drowsiness-yolo-updated/DD_updated'
RUNS_DIR       = '/kaggle/working/runs'
test_img_dir   = f"{DATASET_PATH}/test/images"
test_label_dir = f"{DATASET_PATH}/test/labels"
class_names    = ['EyeClosed', 'Yawn', 'EyeOpen']
device         = next(model_fcbam_best.model.parameters()).device

raw_model = model_fcbam_best.model
raw_model = raw_model.to(device)
raw_model.eval()


class GradCAM:
    def __init__(self, model, layer_idx):
        self.model       = model
        self.layer_idx   = layer_idx
        self.activations = None
        self.gradients   = None

        target = model.model[layer_idx]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self.activations = output.clone()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone()

    def generate(self, img_tensor):
        self.activations = None
        self.gradients   = None
        self.model.zero_grad()

        with torch.enable_grad():
            x    = img_tensor.clone().requires_grad_(True)

            # manually iterate layers - bypasses any inference_mode
            # decorator on model.forward()
            y    = []
            feat = x
            for i, layer in enumerate(self.model.model):
                if hasattr(layer, 'f') and layer.f != -1:
                    if isinstance(layer.f, int):
                        feat = y[layer.f]
                    else:
                        feat = [feat if j == -1 else y[j] for j in layer.f]
                feat = layer(feat)
                y.append(feat if i in self.model.save else None)

            # feat = final Detect output, a tensor not a dict in eval mode
            if isinstance(feat, (list, tuple)):
                score = feat[0].sum()
            else:
                score = feat.sum()

            score.backward()

        if self.activations is None or self.gradients is None:
            print("    hooks not triggered")
            return None

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().detach().numpy()

        if cam.max() == cam.min():
            print("    cam is flat")
            return None

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img_path, cam, alpha=0.45):
    img     = cv2.imread(img_path)
    img     = cv2.resize(img, (640, 640))
    cam_r   = cv2.resize(cam, (640, 640))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


gradcam = GradCAM(raw_model, layer_idx=19)

# ── Pick 2 images per class ───────────────────────────────────────────────────
image_paths  = sorted(glob.glob(f"{test_img_dir}/*.jpg") +
                      glob.glob(f"{test_img_dir}/*.png"))

class_images = {0: [], 1: [], 2: []}
for img_path in image_paths:
    label_path = os.path.join(test_label_dir, Path(img_path).stem + '.txt')
    gt         = load_ground_truth(label_path)
    if len(gt) > 0 and len(class_images[gt[0]]) < 2:
        class_images[gt[0]].append((img_path, gt[0]))

selected = []
for cls_idx in range(3):
    selected.extend(class_images[cls_idx])

print(f"Selected {len(selected)} images")


fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('GradCAM Visualization - YOLOv11s-FCBAM\n'
             'Target Layer: FCBAM-3 (P3, Layer 19)',
             fontsize=14, fontweight='bold')

row = 0
for pair_idx in range(0, len(selected), 2):
    pair = selected[pair_idx:pair_idx+2]
    for col_offset, (img_path, gt_cls) in enumerate(pair):

        img_pil    = Image.open(img_path).convert('RGB').resize((640, 640))
        img_np     = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to(device)

        try:
            cam = gradcam.generate(img_tensor)

            if cam is None:
                continue

            orig, overlay = overlay_cam(img_path, cam)

            col_orig    = col_offset * 2
            col_overlay = col_offset * 2 + 1

            axes[row, col_orig].imshow(img_np)
            axes[row, col_orig].set_title(f"{class_names[gt_cls]} - Original",
                                           fontsize=9, fontweight='bold')
            axes[row, col_orig].axis('off')

            axes[row, col_overlay].imshow(overlay)
            axes[row, col_overlay].set_title(f"{class_names[gt_cls]} - GradCAM",
                                              fontsize=9, fontweight='bold')
            axes[row, col_overlay].axis('off')

        except Exception as e:
            print(f"  Error: {e}")
            continue

    row += 1

raw_model.eval()

plt.tight_layout()
plt.savefig(f'{RUNS_DIR}/YOLOv11s_FCBAM/gradcam_fcbam.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("GradCAM saved")
