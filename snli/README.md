### Baseline Results (dev / test)
|          |28k           |59k           |120k
|----------|--------------|--------------|-------------
|V=20000   |62.17 / 61.25 |66.74 / 66.99 |71.12 / 70.56

### LSTM-LVM (vMF)
##### code_dim=50, enc_dropout=0.5, disc_dropout=0.1, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k          
|------|----|------|--------------|--------------|-------------
|100   |0.5 |23.57 |65.73 / 65.43 |66.67 / 66.83 |68.70 / 68.92
|100   |0.8 |23.57 |65.83 / 65.05 |66.88 / 67.25 |68.63 / 68.32
|100   |1.0 |23.57 |65.46 / 65.68 |68.27 / 68.63 |70.62 / 70.42
|120   |0.5 |27.09 |65.67 / 65.64 |67.23 / 67.06 |68.91 / 69.13
|120   |0.8 |27.09 |66.64 / 66.82 |**68.60 / 68.74** |69.68 / 70.14
|120   |1.0 |31.60 |65.58 / 65.81 |67.82 / 67.80 |69.77 / 69.70
|150   |0.5 |31.60 |65.73 / 65.20 |66.83 / 67.29 |68.86 / 68.57
|150   |0.8 |31.60 |66.13 / 65.96 |68.19 / 68.54 |69.22 / 69.45
|150   |1.0 |31.60 |**66.71 / 65.57** |67.79 / 68.06 |**71.15 / 71.08**

##### code_dim=150, enc_dropout=0.5, disc_dropout=0.1, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k          
|------|----|------|--------------|--------------|-------------
|100   |0.5 |21.62 |63.58 / 63.33 |64.73 / 64.70 |65.24 / 65.11
|100   |0.8 |21.62 |64.32 / 64.45 |66.72 / 66.20 |67.56 / 67.66
|100   |1.0 |21.62 |64.48 / 64.65 |66.03 / 66.15 |68.28 / 67.86
|120   |0.5 |27.59 |64.35 / 65.38 |66.36 / 66.06 |65.61 / 65.90
|120   |0.8 |27.59 |65.04 / 64.58 |66.54 / 66.44 |67.48 / 67.64
|120   |1.0 |27.59 |65.24 / 65.29 |66.54 / 66.39 |68.76 / 68.68
|150   |0.5 |36.17 |65.00 / 64.87 |65.40 / 65.70 |66.08 / 65.95
|150   |0.8 |36.17 |64.99 / 65.01 |66.20 / 66.03 |67.59 / 67.78
|150   |1.0 |36.17 |65.53 / 65.02 |66.91 / 67.36 |67.64 / 67.70

### Naive vMF / shared, tied (V=20000)
##### code_dim=50, enc_dropout=0.25, disc_dropout=0.0, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k          |all
|------|----|------|--------------|--------------|--------------|---
|50    |0.2 |12.11 |62.98 / 63.12 |67.95 / 68.42 |70.11 / 69.77 |
|80    |0.2 |19.52 |65.87 / 66.42 |? / ?         |71.73 / 72.15 |
|100   |0.2 |23.57 |66.87 / 66.62 |70.65 / 70.30 |70.11 / 69.43 |
|50    |0.5 |12.11 |65.48 / 65.75 |? / ?         |76.64 / 76.21 |
|80    |0.5 |19.52 |66.21 / 67.63 |? / ?         |76.88 / 75.99 |
|100   |0.5 |23.57 |66.08 / 67.43 |68.64 / 69.57 |76.65 / 76.36 |
|50    |0.8 |12.11 |64.49 / 64.68 |71.59 / 71.60 |76.46 / 75.66 |
|80    |0.8 |19.52 |65.50 / 65.99 |72.66 / 72.30 |77.09 / 75.79 |
|100   |0.8 |23.57 |66.22 / 66.91 |72.15 / 71.99 |77.19 / 76.36 |
|50    |1.0 |12.11 |64.53 / 64.70 |71.32 / 71.99 |76.71 / 75.70 |83.72 / 82.43
|80    |1.0 |19.52 |65.32 / 66.00 |72.37 / 71.57 |76.85 / 76.05 |84.17 / 83.21
|100   |1.0 |23.57 |65.33 / 65.99 |72.14 / 72.56 |77.34 / 76.36 |83.65 / 82.73

##### code_dim=50, enc_dropout=0.25, disc_dropout=0.1, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k
|------|----|------|--------------|--------------|-------------
|50    |0.1 |12.11 |52.42 / 52.70 |52.21 / 53.30 |61.14 / 61.97
|80    |0.1 |19.52 |58.35 / 58.86 |58.16 / 59.00 |61.99 / 63.13
|100   |0.1 |23.57 |53.84 / 54.09 |58.33 / 59.07 |62.17 / 62.29
|50    |0.2 |12.11 |62.47 / 63.63 |68.12 / 67.86 |68.89 / 69.17
|80    |0.2 |19.52 |64.52 / 65.55 |68.68 / 69.26 |69.39 / 69.35
|100   |0.2 |23.57 |66.64 / 66.95 |68.16 / 68.72 |71.04 / 70.99
|50    |0.5 |12.11 |66.23 / 66.83 |72.03 / 71.42 |75.77 / 75.48
|80    |0.5 |19.52 |66.40 / 67.12 |72.48 / 72.91 |74.87 / 74.62
|100   |0.5 |23.57 |66.78 / 66.82 |72.58 / 72.13 |76.92 / 76.65
|50    |0.8 |12.11 |64.66 / 65.24 |72.18 / 71.73 |77.01 / 76.09
|80    |0.8 |19.52 |65.94 / 65.61 |72.87 / 72.83 |77.09 / 75.76
|100   |0.8 |23.57 |65.97 / 66.17 |73.20 / 72.64 |77.53 / 76.47
|50    |1.0 |12.11 |64.48 / 64.78 |71.92 / 71.56 |76.80 / 75.15
|80    |1.0 |19.52 |65.68 / 66.13 |72.20 / 72.62 |77.48 / 76.70
|100   |1.0 |23.57 |65.74 / 65.56 |72.73 / 72.37 |77.62 / 76.63

##### code_dim=50, enc_dropout=0.25, disc_dropout=0.2, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k
|------|----|------|--------------|--------------|-------------
|50    |0.2 |12.11 |63.51 / 64.19 |66.66 / 67.01 |67.50 / 68.10
|80    |0.2 |19.52 |64.05 / 63.84 |67.29 / 67.49 |68.60 / 69.07
|100   |0.2 |23.57 |61.84 / 62.34 |67.12 / 68.26 |70.82 / 70.39
|50    |0.5 |12.11 |65.35 / 65.57 |71.92 / 72.24 |74.87 / 74.44
|80    |0.5 |19.52 |66.81 / 67.00 |72.42 / 72.17 |74.86 / 74.30
|100   |0.5 |23.57 |66.25 / 67.51 |73.07 / 72.40 |75.68 / 75.11
|50    |0.8 |12.11 |65.30 / 65.88 |71.59 / 71.73 |76.50 / 76.09
|80    |0.8 |19.52 |66.12 / 66.82 |72.76 / 72.17 |76.54 / 75.62
|100   |0.8 |23.57 |65.99 / 66.94 |73.12 / 72.52 |77.32 / 76.50
|50    |1.0 |12.11 |63.35 / 64.27 |72.08 / 72.26 |76.67 / 76.32
|80    |1.0 |19.52 |65.03 / 66.08 |72.68 / 72.01 |77.15 / 76.24
|100   |1.0 |23.57 |64.89 / 66.05 |72.55 / 71.50 |77.27 / 76.21

##### code_dim=50, enc_dropout=0.50, disc_dropout=0.1, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k
|------|----|------|--------------|--------------|-------------
|50    |0.2 |12.11 |63.53 / 64.12 |67.07 / 67.61 |69.54 / 69.92
|80    |0.2 |19.52 |63.54 / 63.70 |67.89 / 68.12 |70.00 / 70.43
|100   |0.2 |23.57 |61.57 / 62.66 |67.36 / 68.30 |70.26 / 70.22
|120   |0.2 |27.09 |65.01 / 65.41 |68.38 / 68.54 |68.10 / 68.29
|150   |0.2 |31.60 |65.34 / 65.78 |68.25 / 68.34 |68.40 / 68.60
|50    |0.5 |12.11 |65.89 / 66.17 |71.99 / 72.14 |75.08 / 75.09
|80    |0.5 |19.52 |67.03 / 67.96 |72.49 / 72.33 |75.62 / 74.93
|100   |0.5 |23.57 |68.25 / 68.37 |72.97 / 73.51 |75.72 / 75.52
|120   |0.5 |27.09 |67.62 / 68.48 |73.14 / 72.64 |75.97 / 75.88
|150   |0.5 |31.60 |67.64 / 68.15 |73.19 / 72.78 |75.98 / 75.63
|50    |0.8 |12.11 |65.49 / 66.49 |72.54 / 72.31 |76.57 / 76.05
|80    |0.8 |19.52 |66.94 / 66.71 |73.81 / 72.93 |76.84 / 76.37
|100   |0.8 |23.57 |66.25 / 66.93 |73.70 / 73.35 |77.77 / 76.61
|120   |0.8 |27.09 |66.78 / 67.12 |73.65 / 73.43 |**78.19 / 76.88**
|150   |0.8 |31.60 |67.09 / 67.49 |73.94 / 73.56 |77.63 / 77.28
|50    |1.0 |12.11 |64.10 / 64.50 |72.15 / 72.04 |76.95 / 76.22
|80    |1.0 |19.52 |66.10 / 66.64 |73.19 / 72.66 |77.49 / 77.05
|100   |1.0 |23.57 |66.09 / 66.91 |73.37 / 72.93 |77.44 / 76.69
|120   |1.0 |27.09 |67.04 / 66.86 |73.38 / 73.45 |77.33 / 76.80
|150   |1.0 |31.60 |66.64 / 67.40 |73.54 / 72.79 |78.06 / 76.97

##### code_dim=50, enc_dropout=0.75, disc_dropout=0.1, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k
|------|----|------|--------------|--------------|-------------
|100   |0.2 |23.57 |60.25 / 61.16 |64.88 / 64.73 |64.47 / 64.79
|120   |0.2 |27.09 |60.46 / 59.74 |63.87 / 64.34 |64.05 / 64.40
|150   |0.2 |31.60 |60.42 / 61.09 |64.35 / 65.14 |65.40 / 65.70
|100   |0.5 |23.57 |67.52 / 67.88 |72.29 / 72.69 |74.15 / 73.72
|120   |0.5 |27.09 |67.58 / 67.82 |71.24 / 71.45 |72.14 / 72.46
|150   |0.5 |31.60 |68.46 / 68.71 |70.98 / 71.30 |72.06 / 72.16
|100   |0.8 |23.57 |68.51 / 68.39 |73.03 / 72.41 |75.88 / 75.34
|120   |0.8 |27.09 |68.07 / 68.48 |72.99 / 72.92 |75.80 / 75.37
|150   |0.8 |31.60 |**68.52 / 68.38** |73.85 / 73.67 |76.56 / 75.50
|100   |1.0 |23.57 |67.30 / 68.40 |**74.12 / 73.53** |77.20 / 76.59
|120   |1.0 |27.09 |67.26 / 67.55 |73.31 / 72.91 |76.74 / 75.96
|150   |1.0 |31.60 |67.42 / 68.55 |73.11 / 73.63 |76.63 / 76.21

##### code_dim=100, enc_dropout=0.25, disc_dropout=0.1, l2_weight=0.002
|kappa |dw  |KLD   |28k           |59k           |120k
|------|----|------|--------------|--------------|-------------
|50    |0.1 |12.11 |50.93 / 51.41 |59.12 / 60.38 |61.13 / 61.38
|80    |0.1 |19.52 |49.13 / 49.31 |59.31 / 60.13 |59.93 / 60.38
|100   |0.1 |23.57 |57.70 / 58.20 |51.18 / 51.00 |60.21 / 60.20
|50    |0.2 |12.11 |62.13 / 62.36 |67.53 / 68.21 |67.93 / 68.80
|80    |0.2 |19.52 |64.99 / 65.23 |69.08 / 69.08 |69.10 / 69.14
|100   |0.2 |23.57 |63.36 / 64.14 |68.51 / 68.90 |69.05 / 69.00
|50    |0.5 |12.11 |64.18 / 64.85 |71.45 / 71.05 |75.15 / 74.26
|80    |0.5 |19.52 |65.50 / 65.73 |72.68 / 72.42 |76.37 / 75.76
|100   |0.5 |23.57 |65.47 / 65.63 |72.60 / 73.23 |76.58 / 76.04
|50    |0.8 |12.11 |61.92 / 61.37 |71.47 / 71.13 |75.96 / 75.78
|80    |0.8 |19.52 |64.59 / 65.47 |73.04 / 72.07 |77.34 / 76.92
|100   |0.8 |23.57 |65.86 / 67.10 |72.75 / 72.60 |77.17 / 75.92
|50    |1.0 |12.11 |62.85 / 63.72 |70.60 / 70.68 |76.18 / 75.32
|80    |1.0 |19.52 |64.34 / 64.76 |72.08 / 72.04 |77.04 / 76.28
|100   |1.0 |23.57 |65.00 / 65.63 |72.40 / 72.10 |77.69 / 76.00
