# Synthetic evaluation (100 docs, seed 0, OCR tesseract)

Overall field accuracy: **0.9267** over 300 fields; 512.1 ms/doc end to end on arm64.

| field | n | accuracy |
|---|---|---|
| vendor_name | 100 | 1.0 |
| invoice_date | 100 | 0.98 |
| total_amount | 100 | 0.8 |

| degradation | n fields | accuracy |
|---|---|---|
| clean | 60 | 1.0 |
| noisy | 60 | 1.0 |
| rotated | 60 | 1.0 |
| degraded | 60 | 0.9833 |
| severe | 60 | 0.65 |

| threshold | flagged fields | share flagged | accuracy auto | accuracy flagged | docs fully auto with an error |
|---|---|---|---|---|---|
| 0.5 | 15 | 0.05 | 0.9719 | 0.0667 | 6/100 |
| 0.6 | 16 | 0.0533 | 0.9754 | 0.0625 | 5/100 |
| 0.7 | 21 | 0.07 | 0.9857 | 0.1429 | 3/100 |
| 0.75 | 26 | 0.0867 | 1.0 | 0.1538 | 0/100 |
| 0.8 | 33 | 0.11 | 1.0 | 0.3333 | 0/100 |
| 0.85 | 141 | 0.47 | 1.0 | 0.844 | 0/100 |
