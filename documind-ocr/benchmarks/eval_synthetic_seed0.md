# Synthetic evaluation (100 docs, seed 0, OCR tesseract)

Overall field accuracy: **0.9233** over 300 fields; 480.2 ms/doc end to end on arm64.

| field | n | accuracy |
|---|---|---|
| vendor_name | 100 | 0.99 |
| invoice_date | 100 | 0.98 |
| total_amount | 100 | 0.8 |

| degradation | n fields | accuracy |
|---|---|---|
| clean | 60 | 1.0 |
| noisy | 60 | 1.0 |
| rotated | 60 | 1.0 |
| degraded | 60 | 0.9833 |
| severe | 60 | 0.6333 |

| threshold | flagged fields | share flagged | accuracy auto | accuracy flagged | docs fully auto with an error |
|---|---|---|---|---|---|
| 0.5 | 16 | 0.0533 | 0.9683 | 0.125 | 6/100 |
| 0.6 | 17 | 0.0567 | 0.9717 | 0.1176 | 5/100 |
| 0.7 | 75 | 0.25 | 0.9778 | 0.76 | 1/100 |
| 0.75 | 78 | 0.26 | 0.991 | 0.7308 | 0/100 |
| 0.8 | 86 | 0.2867 | 1.0 | 0.7326 | 0/100 |
| 0.85 | 141 | 0.47 | 1.0 | 0.8369 | 0/100 |
