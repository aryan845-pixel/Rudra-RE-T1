# Day09 Notes – Object Detection (Hinglish notes)

## 1. Object Detection kya hai?
Image me **kya** hai (class) aur **kaha** hai (bounding box) – dono batana. Classification sirf "kya" batata hai; detection "kya + kaha + kitni baar".

Ek detection = `class label` + `confidence (0–1)` + `box (x1,y1,x2,y2)`.

## 2. Zaroori terms
| Term | Matlab |
|------|--------|
| Bounding box | Object ke around rectangle; top-left (x1,y1), bottom-right (x2,y2) |
| Confidence | Model kitna sure hai. Threshold (0.25) se kam wale drop |
| IoU | Intersection over Union = overlap area / total area of 2 boxes (0 se 1) |
| NMS | Non-Max Suppression: ek hi object ke duplicate boxes me se best rakho, baaki hatao (IoU zyada ho to) |
| COCO | 80-class dataset jispe YOLOv8n pretrained hai (person, car, dog, ...) |
| Pretrained | Pehle se train hua; humein dobara train nahi karna |
| Inference | Trained model se prediction chalana |

## 3. YOLO kaise kaam karta hai (simple)
"You Only Look Once": image ek hi baar network se jaati hai, output me saare boxes + class scores aate hain. Isliye fast hai. Nano (`n`) = sabse chhota, CPU pe chalne layak.

## 4. Torch vs ONNX Runtime
- **Torch/Ultralytics:** easy API (`model.predict`), weights auto-download. Install bhaari (torch).
- **ONNX Runtime:** halka, deploy-friendly, par pre/post-processing (resize, NMS) khud likhna padta hai.
- Humne **Ultralytics (Torch)** chuna kyunki simple + brief me bhi allowed.

## 5. Pipeline (code flow)
1. `load_image()` – path check, file read, `cv2.imdecode` (missing/corrupt -> InputError, exit 2)
2. `load_model()` – ultralytics import + YOLO load (fail -> ModelError, exit 3)
3. `run_inference()` – `model.predict(conf=...)`, time `perf_counter` se
4. `draw_detections()` – `cv2.rectangle` + `putText`, har class ka fixed colour
5. `save_image()` + `build_report()` -> `detected.jpg`, `run_report.json`
6. `main()` – exceptions ko exit codes me map karta hai

## 6. Commands
```bash
pip install -r requirements.txt
python object_detection.py --fetch-sample          # sample photo + run
python object_detection.py --image images/test.jpg --confidence 0.25
python object_detection.py --confidence 0.9        # zero-detection test (T-04)
python -m unittest -v tests.test_input_validation
```

## 7. Confidence threshold ka effect
Kam (0.1) -> zyada boxes, zyada false positives. Zyada (0.9) -> sirf pakke objects, kuch miss. Default 0.25 balanced demo ke liye.

## 8. Common problems
- `MODEL ERROR: ultralytics not installed` -> `pip install -r requirements.txt`
- Weights download fail (offline) -> net on karke ek baar run karo, ya `yolov8n.pt` manually download karke `--model path` do
- Koi detection nahi -> threshold kam karo / real photo use karo (synthetic placeholder me objects nahi)

## 9. Viva / interview questions
1. Classification aur detection me kya farak hai?
2. IoU kya hai, NMS kyu chahiye?
3. Confidence threshold badhane se kya hota hai?
4. YOLO single-stage kyu kehlate hain?
5. ONNX Runtime kab prefer karoge?
6. Humne speed/accuracy claim kyu nahi ki? (Brief: sirf measured values likho)

## 10. Aage: Day10 – Semantic Segmentation
Detection box deta hai; segmentation **har pixel** ko class deta hai (e.g. road, sky, car).
