from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import os

app = FastAPI(title="Skin Lesion Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_FILES = {
    # keys are internal ids; display names will be provided in results
    "resnet50": os.path.join(os.path.dirname(__file__), "..", "common", "ResNet50_Finetuned_best.pth"),
    "mobilenet_v3_large": os.path.join(os.path.dirname(__file__), "..", "common", "MobileNetV3-Large_best.pth"),
    "efficientnet_b0": os.path.join(os.path.dirname(__file__), "..", "common", "EfficientNet-B0_best.pth"),
    "densenet121": os.path.join(os.path.dirname(__file__), "..", "common", "final_skin_disease_model.pth"),
}

# Friendly display names for the frontend
DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "Densenet121",
}


def default_transform():
    # Standard ImageNet-style preprocessing; most models expect 224x224
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def pil_to_tensor(img: Image.Image):
    # convert PIL image to tensor of shape [1,C,H,W]
    if img.mode != "RGB":
        img = img.convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    t = tf(img)
    # normalize separately (so ToTensor keeps values 0-1)
    t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)
    return t.unsqueeze(0)


def try_load_model(path, name):
    device = torch.device("cpu")
    if not os.path.exists(path):
        return None, f"file not found: {path}"

    try:
        obj = torch.load(path, map_location=device)
    except Exception as e:
        return None, f"torch.load failed: {e}"

    # If the saved object is a full module
    if isinstance(obj, torch.nn.Module):
        model = obj
        model.to(device).eval()
        return model, None

    # If it's a state_dict (dict)
    if isinstance(obj, dict):
        # handle wrapped checkpoints saved by training notebooks
        ckpt = obj
        state = obj
        # unwrap common wrapper keys
        if 'model_state_dict' in obj:
            state = obj['model_state_dict']
        # prefer explicit num_classes if present in wrapper
        ckpt_num_classes = None
        if isinstance(ckpt, dict) and 'num_classes' in ckpt:
            try:
                ckpt_num_classes = int(ckpt.get('num_classes'))
            except Exception:
                ckpt_num_classes = None

        # try to infer number of output classes from checkpoint keys if not provided
        num_classes = ckpt_num_classes
        if num_classes is None:
            if isinstance(state, dict):
                if 'fc.weight' in state:
                    num_classes = state['fc.weight'].shape[0]
                elif 'classifier.3.weight' in state:
                    num_classes = state['classifier.3.weight'].shape[0]
                elif 'classifier.1.weight' in state:
                    num_classes = state['classifier.1.weight'].shape[0]
                elif 'classifier.weight' in state:
                    num_classes = state['classifier.weight'].shape[0]

        try:
            # Instantiate and adapt model according to inferred architecture name
            lname = name.lower()

            if 'resnet' in lname or 'resnet50' in lname:
                model = models.resnet50(weights=None)
                if num_classes is not None:
                    num_ftrs = model.fc.in_features
                    model.fc = torch.nn.Linear(num_ftrs, int(num_classes))
                model.to(device)
                model.load_state_dict(state, strict=False)
                model.eval()
                return model, None

            if 'densenet' in lname:
                model = models.densenet121(weights=None)
                if num_classes is not None:
                    num_ftrs = model.classifier.in_features
                    # match training notebook: Sequential(Dropout(0.3), Linear(num_ftrs, num_classes))
                    model.classifier = torch.nn.Sequential(
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(num_ftrs, int(num_classes))
                    )
                model.to(device)
                model.load_state_dict(state, strict=False)
                model.eval()
                return model, None

            if 'efficientnet' in lname or 'efficientnet_b0' in lname:
                model = models.efficientnet_b0(weights=None)
                if num_classes is not None:
                    # classifier is Sequential(Dense, Linear) in many torchvision versions
                    try:
                        num_ftrs = model.classifier[1].in_features
                        model.classifier[1] = torch.nn.Linear(num_ftrs, int(num_classes))
                    except Exception:
                        # fallback to replace whole classifier
                        try:
                            num_ftrs = model.classifier[1].in_features
                        except Exception:
                            num_ftrs = None
                        if num_ftrs is not None:
                            model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(num_ftrs, int(num_classes)))
                model.to(device)
                model.load_state_dict(state, strict=False)
                model.eval()
                return model, None

            # mobilenet variants: prefer v3 large if name indicates it or checkpoint has classifier.3
            if 'mobilenet_v3' in lname or 'mobilenet_v3_large' in lname or 'mobile' in lname:
                # try mobilenet_v3_large first
                try:
                    model = models.mobilenet_v3_large(weights=None)
                    if num_classes is not None:
                        # mobilenet_v3_large classifier has last Linear at classifier[3]
                        try:
                            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, int(num_classes))
                        except Exception:
                            # older versions may have classifier[1]
                            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, int(num_classes))
                    model.to(device)
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    return model, None
                except Exception:
                    # fallback to mobilenet_v2
                    model = models.mobilenet_v2(weights=None)
                    if num_classes is not None:
                        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, int(num_classes))
                    model.to(device)
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    return model, None

            # fallback: try efficientnet
            model = models.efficientnet_b0(weights=None)
            if num_classes is not None:
                try:
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = torch.nn.Linear(num_ftrs, int(num_classes))
                except Exception:
                    pass
            model.to(device)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model, None
        except Exception as e:
            return None, f"failed to instantiate/load arch: {e}"

    return None, "unsupported model file format"


@app.on_event("startup")
def load_models():
    app.state.models = {}
    # try to load class name mapping from common folder (json or txt)
    app.state.label_map = None
    try:
        base_common = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common'))
        json_path = os.path.join(base_common, 'class_names.json')
        txt_path = os.path.join(base_common, 'class_names.txt')
        if os.path.exists(json_path):
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # support either list (index->name) or dict (str(index)->name)
                if isinstance(data, list):
                    app.state.label_map = {i: name for i, name in enumerate(data)}
                elif isinstance(data, dict):
                    # convert keys to int when possible
                    m = {}
                    for k, v in data.items():
                        try:
                            m[int(k)] = v
                        except Exception:
                            pass
                    if m:
                        app.state.label_map = m
        elif os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                app.state.label_map = {i: name for i, name in enumerate(lines)}
    except Exception as e:
        print(f"Failed to load label map: {e}")
    for display_name, path in MODEL_FILES.items():
        # normalize to absolute
        path = os.path.abspath(path)
        model, err = try_load_model(path, display_name)
        if model is None:
            app.state.models[display_name] = {"loaded": False, "error": err}
            print(f"Model {display_name} failed to load: {err}")
        else:
            app.state.models[display_name] = {"loaded": True, "model": model}
            print(f"Model {display_name} loaded successfully")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"invalid image: {e}"})

    input_tensor = pil_to_tensor(img)

    results = []

    for name, info in app.state.models.items():
        if not info.get("loaded"):
            results.append({"model": DISPLAY_NAMES.get(name, name), "error": info.get("error")})
            continue

        model = info.get("model")
        try:
            with torch.no_grad():
                out = model(input_tensor)
                # many models return logits; ensure tensor
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if not torch.is_tensor(out):
                    # try to convert
                    out = torch.tensor(out)
                probs = F.softmax(out, dim=1)
                top1_prob, top1_idx = torch.max(probs, dim=1)
                top1_idx = int(top1_idx.item())
                top1_prob = float(top1_prob.item())
                # map index to human label if available
                label_name = None
                if getattr(app.state, 'label_map', None) is not None:
                    label_name = app.state.label_map.get(top1_idx)

                if label_name:
                    label = label_name
                else:
                    label = f"class_{top1_idx}"

                results.append({
                    "model": DISPLAY_NAMES.get(name, name),
                    "label": label,
                    "confidence": round(top1_prob, 4)
                })
        except Exception as e:
            results.append({"model": name, "error": f"inference error: {e}"})

    # Return also a tiny preview size and mime type? We'll just return results
    return {"results": results}


@app.get("/labels")
async def get_labels():
    """Return the loaded label map (index -> name) if available."""
    lm = getattr(app.state, 'label_map', None)
    if lm is None:
        return {"loaded": False, "message": "No label map found. Place common/class_names.json or common/class_names.txt"}
    # convert keys to strings for JSON serializable
    return {"loaded": True, "labels": {str(k): v for k, v in lm.items()}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
