# Daily Engineering Log — Day 10
(Draft: fixed-format template from your PDF was not available here — paste these fields into your existing template.)

- **Date:** 14 October 2026
- **Assigned Task:** Image Denoising & Enhancement
- **What Was Studied:** Gaussian, median aur bilateral filtering; percentile contrast stretching; gamma correction aur CLAHE for low-light images.
- **What Was Implemented:** Python/OpenCV pipeline (validation → denoise → stretch/low-light → comparison → metrics → saved outputs), synthetic sample generator, CLI, 12 automated tests, README.
- **Tools Used:** Python, OpenCV, NumPy, scikit-image (SSIM), unittest.
- **Output Produced:** original.png, denoised.png, enhanced.png, comparison.png, run_summary.json, quality_metrics.json for 4 synthetic samples (gaussian noise, salt-and-pepper, low contrast, low light).
- **Problems Encountered:** Pehle low-light default (gamma 0.6, CLAHE clip 2.0) output ko over-bright aur blotchy bana raha tha. pytest build environment me installed nahi tha.
- **How Problems Were Resolved:** Few settings try karke milder defaults chose kiye (gamma 0.8, clip 1.5, bilateral sigma_color 15 for low-light sample). Tests ko unittest me likha jo pytest ke saath bhi chalte hain.
- **Actual Test Results:** 12 / 12 passed. Denoising PSNR vs clean: gaussian-noise 20.39→27.58 dB, salt-and-pepper 17.50→24.03 dB. Low-light PSNR 7.00→15.23 dB. Contrast stretch ne PSNR vs clean thoda kam kiya (low_contrast 14.77→13.96 dB) — contrast badhna = quality better nahi.
- **Next Task:** Super-Resolution Study
- **Personal Expenditure:** ₹0
