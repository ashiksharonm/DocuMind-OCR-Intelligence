# Synthetic evaluation (100 docs, seed 1, OCR tesseract)

Overall field accuracy: **0.9367** over 300 fields; 505.6 ms/doc end to end on arm64.

| field | n | accuracy |
|---|---|---|
| vendor_name | 100 | 1.0 |
| invoice_date | 100 | 0.98 |
| total_amount | 100 | 0.83 |

| degradation | n fields | accuracy |
|---|---|---|
| clean | 60 | 1.0 |
| noisy | 60 | 1.0 |
| rotated | 60 | 1.0 |
| degraded | 60 | 0.9667 |
| severe | 60 | 0.7167 |

| threshold | flagged fields | share flagged | accuracy auto | accuracy flagged | docs fully auto with an error |
|---|---|---|---|---|---|
| 0.5 | 10 | 0.0333 | 0.9655 | 0.1 | 9/100 |
| 0.6 | 12 | 0.04 | 0.9722 | 0.0833 | 8/100 |
| 0.7 | 72 | 0.24 | 0.9781 | 0.8056 | 1/100 |
| 0.75 | 78 | 0.26 | 0.991 | 0.7821 | 1/100 |
| 0.8 | 88 | 0.2933 | 1.0 | 0.7841 | 0/100 |
| 0.85 | 147 | 0.49 | 1.0 | 0.8707 | 0/100 |
