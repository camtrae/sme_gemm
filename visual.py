#!/usr/bin/env python3
"""
SME Matrix Multiplication Performance Visualization Script
ARM Scalable Matrix Extension Optimization Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {
        'cpu': '#7F7F7F',
        'sme_cpu_prep': '#045DB7',
        'sme_sme_prep': '#6A178B',
        'sme_4tiles': '#B580C5',
        'grid': '#E8E8E8',
        'text': '#333333',
        'baseline': '#7F7F7F',
    }
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.linewidth': 1.5,
        'axes.edgecolor': colors['text'],
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    return colors

def load_data():
    matrix_sizes = ['64×64×64', '128×128×128', '256×256×256']
    
    time_data = {
        'cpu': [453.0, 2985.9, 11554.4],
        'sme_cpu_prep': [39.4, 59.6, 140.2],
        'sme_sme_prep': [5.1, 18.8, 66.1],
        'sme_4tiles': [5.3, 8.4, 23.0],
    }
    
    speedup_data = {
        'cpu': [1.00, 1.00, 1.00],
        'sme_cpu_prep': [11.50, 50.10, 82.41],
        'sme_sme_prep': [88.82, 158.82, 174.80],
        'sme_4tiles': [85.47, 355.46, 502.37],
    }
    
    gflops_data = {
        'cpu': [1.16, 1.40, 2.90],
        'sme_cpu_prep': [13.31, 70.37, 239.33],
        'sme_sme_prep': [102.80, 223.10, 507.63],
        'sme_4tiles': [98.92, 499.32, 1458.89],
    }
    
    return matrix_sizes, time_data, speedup_data, gflops_data

def create_speedup_plot(ax, matrix_sizes, speedup_data, colors):
    x = np.arange(len(matrix_sizes))
    
    methods = {
        'sme_cpu_prep': ('SME (CPU Transpose + Single Tile)', 'o', colors['sme_cpu_prep']),
        'sme_sme_prep': ('SME (SME Transpose + Single Tile)', 's', colors['sme_sme_prep']),
        'sme_4tiles': ('SME (SME Transpose + 4-Tiles Parallel)', 'D', colors['sme_4tiles']),
    }
    
    for key, (label, marker, color) in methods.items():
        ax.plot(x, speedup_data[key], marker=marker, color=color, 
               linewidth=3.0, markersize=10, markeredgewidth=2,
               markeredgecolor='white', label=label, alpha=0.9, zorder=3)
    
    for i, val in enumerate(speedup_data['sme_4tiles']):
        y_offset = val * 1.10 if i != 1 else val * 1.15
        ax.text(x[i], y_offset, f'{val:.1f}×', ha='center', va='bottom',
               fontsize=11, color=colors['sme_4tiles'], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.8, edgecolor=colors['sme_4tiles'], linewidth=0.5))
    
    ax.axhline(y=1.0, color=colors['baseline'], linestyle='--', 
              linewidth=2.0, alpha=0.5, label='Baseline (CPU)', zorder=1)
    
    ax.set_ylabel('Speedup Factor (×)', fontweight='bold', color=colors['text'])
    ax.set_ylim([0.5, 600])
    ax.set_xlim([-0.3, len(matrix_sizes) - 0.7])
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=12)
    ax.set_xlabel('Matrix Size (M×K×N)', fontweight='bold', color=colors['text'])
    ax.set_title('Speedup Factor Analysis', fontweight='bold', pad=15, color=colors['text'])
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11,
             fancybox=False, edgecolor=colors['grid'], ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0, color=colors['grid'])

def create_throughput_plot(ax, matrix_sizes, gflops_data, colors):
    x = np.arange(len(matrix_sizes))
    bar_width = 0.2
    
    methods = [
        ('cpu', 'CPU', colors['cpu'], -1.5),
        ('sme_cpu_prep', 'CPU Prep\n+ Single', colors['sme_cpu_prep'], -0.5),
        ('sme_sme_prep', 'SME Prep\n+ Single', colors['sme_sme_prep'], 0.5),
        ('sme_4tiles', 'SME Prep\n+ 4-Tiles', colors['sme_4tiles'], 1.5),
    ]
    
    for key, label, color, offset in methods:
        values = gflops_data[key]
        bars = ax.bar(x + offset * bar_width, values, bar_width,
                     label=label, color=color, alpha=0.9, 
                     edgecolor='white', linewidth=1.5)
        
        if key == 'sme_4tiles':
            for i, (bar, val) in enumerate(zip(bars, values)):
                if i % 2 == 1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=10,
                           color=color, fontweight='bold')
    
    ax.set_ylabel('Throughput (GFLOPS)', fontweight='bold', color=colors['text'])
    ax.set_ylim([0, 1700])
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=12)
    ax.set_xlabel('Matrix Size (M×K×N)', fontweight='bold', color=colors['text'])
    ax.set_title('Throughput Performance Comparison', fontweight='bold', pad=15, color=colors['text'])
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10.5,
             fancybox=False, edgecolor=colors['grid'], ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', color=colors['grid'])

def create_execution_time_plot(ax, matrix_sizes, time_data, colors):
    x = np.arange(len(matrix_sizes))
    
    methods = {
        'cpu': ('CPU Baseline', 'o', colors['cpu']),
        'sme_cpu_prep': ('CPU Prep + Single Tile', 's', colors['sme_cpu_prep']),
        'sme_sme_prep': ('SME Prep + Single Tile', '^', colors['sme_sme_prep']),
        'sme_4tiles': ('SME Prep + 4-Tiles Parallel', 'D', colors['sme_4tiles']),
    }
    
    for key, (label, marker, color) in methods.items():
        ax.plot(x, time_data[key], marker=marker, color=color, 
               linewidth=3.0, markersize=10, markeredgewidth=2,
               markeredgecolor='white', label=label, alpha=0.9, zorder=3)
    
    best_vals = time_data['sme_4tiles']
    for i in [0, -1]:
        y_offset = best_vals[i] * 0.6 if i == 0 else best_vals[i] * 1.5
        va = 'top' if i == 0 else 'bottom'
        ax.text(x[i], y_offset, f'{best_vals[i]:.1f}μs', ha='center', va=va,
               fontsize=10, color=colors['sme_4tiles'], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.8, edgecolor=colors['sme_4tiles'], linewidth=0.5))
    
    ax.set_ylabel('Execution Time (μs)', fontweight='bold', color=colors['text'])
    ax.set_yscale('log')
    ax.set_ylim([3, 20000])
    ax.set_xlim([-0.3, len(matrix_sizes) - 0.7])
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=12)
    ax.set_xlabel('Matrix Size (M×K×N)', fontweight='bold', color=colors['text'])
    ax.set_title('Execution Time Comparison (Log Scale)', fontweight='bold', pad=15, color=colors['text'])
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11,
             fancybox=False, edgecolor=colors['grid'])
    ax.grid(True, alpha=0.3, linestyle='--', which='both', color=colors['grid'])

def main():
    print("=" * 70)
    print("SME Matrix Multiplication Performance Visualization")
    print("=" * 70)
    
    colors = setup_plot_style()
    matrix_sizes, time_data, speedup_data, gflops_data = load_data()
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    print("\n📊 Generating performance visualizations...")
    create_speedup_plot(ax1, matrix_sizes, speedup_data, colors)
    print("  ✓ Speedup analysis chart created")
    
    create_throughput_plot(ax2, matrix_sizes, gflops_data, colors)
    print("  ✓ GFLOPS comparison chart created")
    
    create_execution_time_plot(ax3, matrix_sizes, time_data, colors)
    print("  ✓ Execution time chart created")
    
    fig.suptitle('ARM SME Matrix Multiplication Performance Analysis', 
                fontsize=20, fontweight='bold', y=1.00, color=colors['text'])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12)
    
    output_formats = {
        'png': {'dpi': 300, 'desc': 'GitHub/Web display (300 DPI)'},
        'pdf': {'dpi': None, 'desc': 'LaTeX/Papers (vector)'},
        'svg': {'dpi': None, 'desc': 'Vector graphics (editable)'},
    }
    
    print("\n💾 Saving visualization files...")
    print("-" * 70)
    
    for fmt, settings in output_formats.items():
        filename = f'sme_matmul_performance.{fmt}'
        plt.savefig(filename, dpi=settings['dpi'], 
                   bbox_inches='tight', facecolor='white')
        print(f'  ✓ {filename:35s} - {settings["desc"]}')
    
    plt.savefig('sme_matmul_performance_hires.png', dpi=600, 
               bbox_inches='tight', facecolor='white')
    print(f'  ✓ {"sme_matmul_performance_hires.png":35s} - Publication quality (600 DPI)')
    
    print("\n" + "=" * 70)
    print("✅ Visualization Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()