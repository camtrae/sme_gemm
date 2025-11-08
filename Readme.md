# ARM SME2 Matrix Multiplication Performance Analysis

High-performance matrix multiplication implementation using ARM Scalable Matrix Extension 2 (SME2) on Apple Silicon M4, demonstrating comprehensive optimization strategies with up to **502× speedup**.

![SME Performance Analysis](sme_matmul_performance_hires.png)

## 🎯 Overview

This project demonstrates the performance benefits of ARM's Scalable Matrix Extension 2 (SME2) for matrix multiplication operations on **Apple Mac mini M4 chip**. It implements and compares four optimization levels:

- **CPU Baseline**: Traditional triple-loop matrix multiplication
- **SME2 + CPU Transpose**: SME2 acceleration with CPU-based matrix preprocessing
- **SME2 + SME2 Transpose**: Fully SME2-accelerated implementation including preprocessing
- **SME2 + 4-Tiles Parallel**: Advanced optimization using all 4 ZA tiles in parallel

## 🖥️ Test Environment

- **Hardware**: Apple Mac mini (2024)
- **Chip**: Apple M4 (ARMv9-A with SME2)
- **Architecture**: arm64
- **Instruction Set**: ARM Scalable Matrix Extension 2 (SME2)
- **Vector Length**: 512-bit (16 × 32-bit floats)

## 📊 Performance Results

Performance comparison on Mac mini M4 (256×256×256 matrices):

| Implementation | Time (μs) | Speedup | GFLOPS | Tile Utilization |
|----------------|-----------|---------|--------|------------------|
| CPU Baseline | 11554.4 | 1.00× | 2.90 | N/A |
| SME2 (CPU Prep + 1) | 140.2 | 82.4× | 239.3 | 25% (1/4) |
| SME2 (SME2 Prep + 1) | 66.1 | 174.8× | 507.6 | 25% (1/4) |
| SME2 (SME2 Prep + 4) | 23.0 | **502.4×** | **1458.9** | **100% (4/4)** |

### Key Findings

- 🚀 **Maximum Speedup**: 502.4× over CPU baseline (256×256×256)
- ⚡ **Peak Throughput**: 1459 GFLOPS on M4
- 📈 **SME2 Transpose Impact**: 2.12× improvement over CPU transpose
- 🔥 **4-Tiles Parallel Impact**: 2.87× improvement over single tile
- ✅ **Full Tile Utilization**: 100% (4/4 ZA tiles active)

## 🚀 Quick Start

### Prerequisites

- **Apple Silicon Mac** with M4 chip
- **macOS Sonoma 14.0+** or newer
- **Xcode Command Line Tools** or **Homebrew GCC 13+**
- **Python 3.8+** with matplotlib for visualization

### Check SME2 Support

```bash
# Check if your Mac supports SME2
sysctl hw.optional.arm.FEAT_SME2

# Should return: hw.optional.arm.FEAT_SME2: 1
```

### Compilation on macOS

```bash
# Using Apple Clang (recommended for M4)
clang -arch arm64 -march=native+sme2 -O3 -o sme_matmul sme_matmul_complete.c -lm

# Or using Homebrew GCC
gcc-13 -march=native+sme2 -O3 -o sme_matmul sme_matmul_complete.c -lm

# Run the benchmark (default: 256×256×256 matrices)
./sme_matmul

# Run with custom matrix size (M K N iterations)
./sme_matmul 512 512 512 10
```

### Compilation Flags Explained

- `-arch arm64`: Target ARM64 architecture (Apple Silicon)
- `-march=native+sme2`: Enable ARMv9-A architecture with SME2 extensions
- `-O3`: Maximum optimization level
- `-lm`: Link math library

## 📈 Visualization

Generate performance comparison charts:

```bash
# Install matplotlib if needed
pip3 install matplotlib numpy

# Generate visualizations
python3 sme_matmul_visualization.py
```

This will create:
- `sme_matmul_performance.png` - Standard resolution (300 DPI)
- `sme_matmul_performance.pdf` - Vector format for papers
- `sme_matmul_performance.svg` - Editable vector graphics
- `sme_matmul_performance_hires.png` - High resolution (600 DPI)

## 📁 Project Structure

```
.
├── sme_matmul_complete.c           # Complete SME2 implementation
├── sme_matmul_visualization.py     # Performance visualization script
├── sme_matmul_performance.png      # Generated performance chart
└── README.md                        # This file
```

## 🔬 Technical Details

### Optimization Strategies

1. **CPU Preprocessing** (`preprocess_left_matrix_cpu`)
   - Traditional CPU-based matrix transpose
   - Prepares data layout for SME2 operations

2. **SME2 Preprocessing** (`preprocess_left_matrix_sme_kernel`)
   - Utilizes SME2 ZA tiles for fast matrix transpose
   - 2.12× faster than CPU preprocessing on M4
   - Uses `svld1_x2`, `svcreate4`, and `svwrite_hor_za32_f32_vg4` intrinsics

3. **Single-Tile Multiplication** (`matmul_sme_kernel`)
   - Uses 1 ZA tile for outer product accumulation
   - `svmopa_za32_m` for fused multiply-accumulate

4. **4-Tiles Parallel** (`matmul_sme_kernel_4tiles`)
   - Processes 4 column blocks simultaneously
   - Full utilization of all 4 available ZA tiles
   - Maximum parallelism and cache efficiency on M4

### Key SME2 Intrinsics Used

- `svmopa_za32_m`: Matrix outer product and accumulate
- `svld1`, `svld1_x2`: Load vectors with predication
- `svzero_za`: Zero ZA tiles
- `svst1_hor_za32`: Store ZA tile rows
- `svwrite_hor_za32_f32_vg4`: Write 4 vectors to ZA
- `svread_ver_za32_f32_vg4`: Read 4 vectors from ZA
- `svwhilelt_b32`, `svwhilelt_c32`: Generate predicates
- `svpsel_lane_b32`, `svpsel_lane_c32`: Predicate lane selection

## 🧪 Testing Different Matrix Sizes

```bash
# 64×64×64 matrices
./sme_matmul 64 64 64 10

# 128×128×128 matrices  
./sme_matmul 128 128 128 10

# 256×256×256 matrices (default)
./sme_matmul 256 256 256 10

# 512×512×512 matrices
./sme_matmul 512 512 512 10
```

## 📊 Performance Scaling

The implementation shows excellent scaling characteristics:

| Matrix Size | CPU (μs) | SME 4-Tiles (μs) | Speedup | GFLOPS |
|-------------|----------|------------------|---------|--------|
| 64³ | 453.0 | 5.3 | 85.5× | 98.9 |
| 128³ | 2985.9 | 8.4 | 355.5× | 499.3 |
| 256³ | 11554.4 | 23.0 | 502.4× | 1458.9 |

## 💡 Tips for Best Performance on M4

1. **Matrix Size**: Use matrix dimensions that are multiples of 16 (M4's vector length)
2. **Memory Alignment**: Ensure matrices are 64-byte aligned for optimal cache performance
3. **Tile Utilization**: For N dimension < 64 (4×16), consider adjusting algorithm for partial tile usage
4. **System Configuration**: Close unnecessary applications for consistent benchmarking
5. **Power Mode**: Keep Mac plugged in for maximum performance

## 🛠️ Troubleshooting

### "System does not support SME"
```bash
# Verify SME2 support on your Mac
sysctl hw.optional.arm.FEAT_SME2

### Compilation Errors
```bash
# If you get "unknown architecture" error:
# Update Xcode Command Line Tools
xcode-select --install

# Or use Homebrew GCC with SME2 support
brew install gcc@13
gcc-13 -march=armv9-a+sme2 -O3 -o sme_matmul sme_matmul_complete.c -lm
```

### Low Performance
```bash
# Check if system is thermal throttling
sudo powermetrics --samplers smc | grep -i "CPU die temperature"

# Ensure high-performance mode (plug in power adapter)
# Close unnecessary applications
# Run benchmark multiple times for warm-up
```

## 🔗 Resources

- [ARM SME2 Specification](https://developer.arm.com/documentation/ddi0602/latest/)
- [ARM C Language Extensions (ACLE)](https://github.com/ARM-software/acle)
- [Apple Silicon Guide](https://developer.apple.com/documentation/apple-silicon)
- [ARM SVE and SME Programming Guide](https://developer.arm.com/documentation/102107/latest/)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This implementation is specifically optimized for Apple Silicon M4 with SME2 support. Performance results may vary on different ARM processors with SME/SME2 capabilities.