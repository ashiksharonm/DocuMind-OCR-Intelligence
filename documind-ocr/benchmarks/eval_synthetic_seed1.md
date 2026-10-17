# Synthetic evaluation (100 docs, seed 1, OCR tesseract)

Overall field accuracy: **0.9367** over 300 fields; 539.3 ms/doc end to end on arm64.

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
| 0.7 | 16 | 0.0533 | 0.9789 | 0.1875 | 6/100 |
| 0.75 | 22 | 0.0733 | 0.9928 | 0.2273 | 2/100 |
| 0.8 | 38 | 0.1267 | 1.0 | 0.5 | 0/100 |
| 0.85 | 147 | 0.49 | 1.0 | 0.8707 | 0/100 |
