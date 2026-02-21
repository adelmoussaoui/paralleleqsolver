import numpy as np
import time
import psutil
import platform
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CEstimator:
    def __init__(self):
        self.results = {}
        self.hardware_info = self.get_hardware_info()
        
    def get_hardware_info(self):
        """Get detailed hardware information"""
        cpu_freq = psutil.cpu_freq()
        return {
            'cpu_model': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'cpu_freq_ghz': cpu_freq.current / 1000 if cpu_freq else 'Unknown',
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__
        }
    
    def equation_simple(self, x):
        """Simple polynomial system: x² - 4 = 0"""
        return x**2 - 4
    
    def equation_trigonometric(self, x):
        """Trigonometric system: sin(x) + x² - 2 = 0"""
        return np.sin(x) + x**2 - 2
    
    def equation_exponential(self, x):
        """Exponential system: exp(x/100) - x/50 - 1 = 0"""
        return np.exp(x/100) - x/50 - 1
    
    def equation_polynomial(self, x):
        """Cubic polynomial system: x³ - 2x² + x - 5 = 0"""
        return x**3 - 2*x**2 + x - 5
    
    def equation_mixed(self, x):
        """Mixed transcendental system"""
        return np.sin(x) + np.exp(x/100) - x**2/1000 - 2
    
    def benchmark_system(self, equation_func, equation_name, sizes=None, num_trials=5, warmup=2):
        """
        Benchmark solving time for different system sizes and estimate C
        
        Parameters:
        - equation_func: The function defining the system
        - equation_name: Name for identification
        - sizes: List of system sizes to test
        - num_trials: Number of trials per size
        - warmup: Number of warmup runs (excluded from timing)
        """
        if sizes is None:
            sizes = [50, 100, 150, 200, 250, 300, 400, 500,1000,2000,5000]
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {equation_name}")
        print(f"Hardware: {self.hardware_info['cpu_model']}")
        print(f"Sizes: {sizes}")
        print(f"Trials: {num_trials}")
        print(f"{'='*60}")
        
        results = {
            'equation_name': equation_name,
            'sizes': sizes,
            'times': [],
            'raw_measurements': [],
            'hardware_info': self.hardware_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Warmup runs
        print("Performing warmup runs...")
        for _ in range(warmup):
            try:
                x0 = np.ones(100) * 1.5
                fsolve(equation_func, x0, full_output=False)
            except:
                pass
        
        # Main benchmarking
        for i, size in enumerate(sizes):
            print(f"Size {i+1}/{len(sizes)}: N={size}", end="")
            
            trial_times = []
            for trial in range(num_trials):
                try:
                    x0 = np.ones(size) * 1.5
                    
                    start_time = time.perf_counter()
                    solution, info, ier, msg = fsolve(equation_func, x0, full_output=True)
                    end_time = time.perf_counter()
                    
                    if ier == 1:  # Successful convergence
                        elapsed = end_time - start_time
                        trial_times.append(elapsed)
                    else:
                        print(f" (convergence failed: {msg})", end="")
                        
                except Exception as e:
                    print(f" (error: {str(e)[:30]}...)")
                    continue
            
            if trial_times:
                median_time = np.median(trial_times)
                results['times'].append(median_time)
                results['raw_measurements'].append({
                    'size': size,
                    'times': trial_times,
                    'median': median_time,
                    'mean': np.mean(trial_times),
                    'std': np.std(trial_times)
                })
                print(f" - {median_time:.6f}s (n={len(trial_times)})")
            else:
                results['times'].append(np.nan)
                results['raw_measurements'].append({'size': size, 'times': [], 'median': np.nan})
                print(f" - FAILED")
        
        # Estimate C using different methods
        C_estimates = self.estimate_C(results)
        results.update(C_estimates)
        
        self.results[equation_name] = results
        return results
    
    def estimate_C(self, results):
        """Estimate C using multiple regression methods"""
        sizes = np.array(results['sizes'])
        times = np.array(results['times'])
        
        # Remove failed measurements
        valid_mask = ~np.isnan(times)
        if np.sum(valid_mask) < 3:
            return {'error': 'Insufficient successful measurements'}
        
        sizes_valid = sizes[valid_mask]
        times_valid = times[valid_mask]
        
        estimates = {}
        
        # Method 1: Direct cubic fit (t = C * N^3)
        try:
            C_direct = times_valid / (sizes_valid ** 3)
            estimates['C_direct_median'] = float(np.median(C_direct))
            estimates['C_direct_mean'] = float(np.mean(C_direct))
            estimates['C_direct_std'] = float(np.std(C_direct))
        except:
            estimates['C_direct_median'] = np.nan
        
        # Method 2: Least squares regression (t = C * N^3)
        try:
            A = np.column_stack([sizes_valid ** 3])
            C_ls, residuals, rank, s = np.linalg.lstsq(A, times_valid, rcond=None)
            estimates['C_least_squares'] = float(C_ls[0])
            estimates['ls_residuals'] = float(residuals[0]) if len(residuals) > 0 else 0.0
        except:
            estimates['C_least_squares'] = np.nan
        
        # Method 3: Log-log regression (log(t) = log(C) + 3*log(N))
        try:
            log_sizes = np.log(sizes_valid)
            log_times = np.log(times_valid)
            slope, intercept = np.polyfit(log_sizes, log_times, 1)
            estimates['C_log_log'] = float(np.exp(intercept))
            estimates['exponent'] = float(slope)  # Should be close to 3
            estimates['r_squared'] = float(np.corrcoef(log_sizes, log_times)[0,1]**2)
        except:
            estimates['C_log_log'] = np.nan
        
        # Method 4: Weighted average (prefer larger sizes as more representative)
        try:
            weights = sizes_valid ** 2  # Weight by size^2
            C_weighted = np.average(times_valid / (sizes_valid ** 3), weights=weights)
            estimates['C_weighted'] = float(C_weighted)
        except:
            estimates['C_weighted'] = np.nan
        
        # Recommended C (prefer weighted average or least squares)
        if not np.isnan(estimates.get('C_weighted')):
            estimates['C_recommended'] = estimates['C_weighted']
        elif not np.isnan(estimates.get('C_least_squares')):
            estimates['C_recommended'] = estimates['C_least_squares']
        else:
            estimates['C_recommended'] = estimates.get('C_direct_median', np.nan)
        
        return estimates
    
    def plot_results(self, equation_name=None):
        """Plot benchmarking results"""
        if equation_name is None:
            equation_name = list(self.results.keys())[0]
        
        results = self.results[equation_name]
        sizes = np.array(results['sizes'])
        times = np.array(results['times'])
        
        valid_mask = ~np.isnan(times)
        sizes_valid = sizes[valid_mask]
        times_valid = times[valid_mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Raw data and cubic fit
        C = results.get('C_recommended', np.nan)
        if not np.isnan(C):
            size_range = np.linspace(min(sizes_valid), max(sizes_valid), 100)
            cubic_fit = C * size_range ** 3
            ax1.plot(size_range, cubic_fit, 'r--', label=f'Cubic fit: t = {C:.2e}·N³', linewidth=2)
        
        ax1.scatter(sizes_valid, times_valid, s=50, alpha=0.7, label='Measured times')
        ax1.set_xlabel('System Size (N)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title(f'Solving Time vs System Size\n{equation_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-log plot
        ax2.loglog(sizes_valid, times_valid, 'bo-', label='Measured times')
        exponent = results.get('exponent', 3)
        ax2.loglog(sizes_valid, results['C_log_log'] * sizes_valid**exponent, 'r--', 
                  label=f'Power law fit: exponent = {exponent:.3f}')
        ax2.set_xlabel('System Size (N)')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Log-Log Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive calibration report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE C ESTIMATION REPORT")
        print("="*80)
        
        print(f"\nHARDWARE INFORMATION:")
        print(f"  CPU: {self.hardware_info['cpu_model']}")
        print(f"  Cores: {self.hardware_info['cpu_cores']} physical, {self.hardware_info['cpu_threads']} logical")
        print(f"  Memory: {self.hardware_info['memory_gb']:.1f} GB")
        print(f"  Platform: {self.hardware_info['platform']}")
        
        for eq_name, results in self.results.items():
            print(f"\n{'='*60}")
            print(f"EQUATION: {eq_name}")
            print(f"{'='*60}")
            
            if 'error' in results:
                print(f"  ERROR: {results['error']}")
                continue
            
            C_rec = results.get('C_recommended', np.nan)
            exponent = results.get('exponent', np.nan)
            r_squared = results.get('r_squared', np.nan)
            
            print(f"  Recommended C: {C_rec:.4e}")
            print(f"  Fitted exponent: {exponent:.4f} (theoretical: 3.000)")
            print(f"  R² of fit: {r_squared:.6f}")
            
            print(f"\n  Alternative Estimates:")
            print(f"    Least Squares:  {results.get('C_least_squares', np.nan):.4e}")
            print(f"    Log-Log:        {results.get('C_log_log', np.nan):.4e}")
            print(f"    Weighted:       {results.get('C_weighted', np.nan):.4e}")
            print(f"    Direct Median:  {results.get('C_direct_median', np.nan):.4e}")
            
            print(f"\n  Measurement Summary:")
            for i, size in enumerate(results['sizes']):
                if i < len(results['times']) and not np.isnan(results['times'][i]):
                    print(f"    N={size:4d}: {results['times'][i]:8.6f}s")
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"c_estimation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename

def main():
    """Main function to run comprehensive C estimation"""
    estimator = CEstimator()  # CORRECTED: Changed from CEstimation to CEstimator
    
    # Define equation systems to benchmark
    equations = [
        (estimator.equation_simple, "Simple Polynomial (x² - 4 = 0)"),
        (estimator.equation_trigonometric, "Trigonometric (sin(x) + x² - 2 = 0)"),
        (estimator.equation_exponential, "Exponential (exp(x/100) - x/50 - 1 = 0)"),
        (estimator.equation_polynomial, "Cubic Polynomial (x³ - 2x² + x - 5 = 0)"),
        (estimator.equation_mixed, "Mixed Transcendental"),
    ]
    
    # Test sizes (adjust based on your hardware capability)
    test_sizes = [500,750,1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]
    
    # Run benchmarks for all equations
    for eq_func, eq_name in equations:
        try:
            estimator.benchmark_system(eq_func, eq_name, sizes=test_sizes, num_trials=3)
        except Exception as e:
            print(f"Failed to benchmark {eq_name}: {e}")
    
    # Generate report and plots
    estimator.generate_report()
    
    # Plot results for each equation
    for eq_name in estimator.results.keys():
        estimator.plot_results(eq_name)
    
    # Save results
    estimator.save_results()
    
    return estimator

if __name__ == "__main__":
    print("Starting comprehensive C estimation for fsolve systems...")
    estimator = main()
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - Recommended C Values")
    print("="*80)
    for eq_name, results in estimator.results.items():
        C_rec = results.get('C_recommended', np.nan)
        if not np.isnan(C_rec):
            print(f"{eq_name:40}: C = {C_rec:.4e}")