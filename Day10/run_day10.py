"""CLI for Day 10. Without --input it generates samples and processes all of them."""
import argparse
import json
from pathlib import Path

import generate_samples
import image_enhancement as ie

HERE = Path(__file__).parent

# (denoise method, denoise params, enhance mode) chosen per synthetic sample
SAMPLE_PLAN = {
    "gaussian_noise":    ("bilateral", {}, "stretch"),
    "salt_pepper_noise": ("median", {}, "stretch"),
    "low_contrast":      ("gaussian", {"ksize": 3}, "stretch"),
    "low_light":         ("bilateral", {"sigma_color": 15.0}, "lowlight"),
}


def main():
    ap = argparse.ArgumentParser(description="Day 10 - Image Denoising & Enhancement")
    ap.add_argument("--input", help="Input image path (default: run all synthetic samples)")
    ap.add_argument("--reference", help="Optional clean reference image (same size) for PSNR/SSIM")
    ap.add_argument("--output-dir", default=str(HERE / "outputs"))
    ap.add_argument("--denoise", default="gaussian", choices=ie.DENOISE_METHODS)
    ap.add_argument("--enhance", default="lowlight", choices=ie.ENHANCE_MODES)
    ap.add_argument("--ksize", type=int, help="Kernel size for gaussian/median")
    args = ap.parse_args()

    if args.input:
        dp = {"ksize": args.ksize} if args.ksize and args.denoise in ("gaussian", "median") else {}
        s = ie.run_pipeline(args.input, args.output_dir, args.denoise, dp, args.enhance,
                            reference_path=args.reference)
        print(json.dumps(s, indent=2))
        return

    paths = generate_samples.generate(HERE / "samples")
    ref = paths["clean_reference"]
    for name, (dm, dp, em) in SAMPLE_PLAN.items():
        s = ie.run_pipeline(paths[name], Path(args.output_dir) / name, dm, dp, em, reference_path=ref)
        m = json.loads(Path(s["output_paths"]["quality_metrics.json"]).read_text())["reference_comparison"]
        print(f"[{name}] {dm}+{em}  {s['processing_time_seconds']}s  "
              f"PSNR orig={m['psnr_original_vs_reference_db']:.2f} -> "
              f"enh={m['psnr_enhanced_vs_reference_db']:.2f} dB")
    print("Outputs in:", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
