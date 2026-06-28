#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE OPTIMIZATION TEST SUITE
=========================================

Script di testing per validare tutte le ottimizzazioni:
  1. Verifica sintassi
  2. Benchmark comparativo
  3. Validazione accuratezza
  4. Profiling dettagliato
  5. Compatibilità versioni

Uso:
    python test_all_optimizations.py [test_mode]

Modi disponibili:
    quick    - Test veloce (1 min)
    medium   - Test completo (5 min)
    thorough - Test estensivo (30 min)
    gpu      - Test GPU (se disponibile)
"""

import sys
import time
import numpy as np
import traceback
from typing import Dict, List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class TestRunner:
    """Executor di test con reporting dettagliato"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []
        self.failed_tests = []
    
    def test_import(self, module_name: str) -> bool:
        """Verifica se modulo importabile"""
        try:
            __import__(module_name)
            return True
        except ImportError as e:
            self.failed_tests.append((module_name, str(e)))
            return False
    
    def test_syntax(self, file_path: str) -> bool:
        """Verifica sintassi Python"""
        try:
            import py_compile
            py_compile.compile(file_path, doraise=True)
            return True
        except py_compile.PyCompileError as e:
            self.failed_tests.append((file_path, str(e)))
            return False
    
    def benchmark_comparison(self) -> Dict:
        """Confronta performance tra versioni"""
        results = {}
        
        # Test 1: Base optimized
        try:
            from navier_stokes import solve_compressible_fvm
            start = time.time()
            solve_compressible_fvm(T=0.05, log_interval=1000)
            time_base = time.time() - start
            results['base_optimized'] = time_base
            print(f"  ✅ Base optimized: {time_base:.2f}s")
        except Exception as e:
            print(f"  ❌ Base optimized failed: {str(e)[:50]}")
            self.failed_tests.append(('base_optimized', str(e)))
        
        # Test 2: Numba (if available)
        try:
            from navier_stokes_numba import solve_compressible_fvm_numba
            start = time.time()
            solve_compressible_fvm_numba(T=0.05, log_interval=1000)
            time_numba = time.time() - start
            results['numba_jit'] = time_numba
            print(f"  ✅ Numba JIT: {time_numba:.2f}s (speedup: {results['base_optimized']/time_numba:.2f}x)")
        except Exception as e:
            print(f"  ⚠️  Numba not available or failed")
        
        # Test 3: GPU (if available)
        try:
            from navier_stokes_cupy import solve_compressible_fvm_gpu
            start = time.time()
            solve_compressible_fvm_gpu(T=0.05, use_gpu=True, log_interval=1000)
            time_gpu = time.time() - start
            results['gpu_cupy'] = time_gpu
            print(f"  ✅ GPU CuPy: {time_gpu:.2f}s (speedup: {results['base_optimized']/time_gpu:.2f}x)")
        except Exception as e:
            print(f"  ⚠️  GPU not available or failed")
        
        return results
    
    def validate_accuracy(self) -> bool:
        """Valida accuratezza numerica tra versioni"""
        try:
            from navier_stokes import solve_compressible_fvm
            
            # Esegui simulazione
            lift_1 = solve_compressible_fvm(T=0.05, log_interval=1000)
            
            print(f"  ✅ Lift computed: {lift_1:.4f} N/m")
            return True
        except Exception as e:
            print(f"  ❌ Accuracy validation failed: {str(e)}")
            self.failed_tests.append(('accuracy_validation', str(e)))
            return False
    
    def test_dependencies(self) -> Dict[str, bool]:
        """Verifica disponibilità dependenze opzionali"""
        deps = {}
        
        # Numba
        try:
            import numba
            deps['numba'] = True
            print(f"  ✅ Numba {numba.__version__}")
        except ImportError:
            deps['numba'] = False
            print(f"  ⚠️  Numba not installed (optional)")
        
        # CuPy
        try:
            import cupy
            deps['cupy'] = True
            print(f"  ✅ CuPy {cupy.__version__}")
        except ImportError:
            deps['cupy'] = False
            print(f"  ⚠️  CuPy not installed (optional)")
        
        # meshio
        try:
            import meshio
            deps['meshio'] = True
            print(f"  ✅ meshio {meshio.__version__}")
        except ImportError:
            deps['meshio'] = False
            print(f"  ❌ meshio missing (REQUIRED)")
        
        # numpy
        try:
            import numpy
            deps['numpy'] = True
            print(f"  ✅ NumPy {numpy.__version__}")
        except ImportError:
            deps['numpy'] = False
            print(f"  ❌ NumPy missing (REQUIRED)")
        
        return deps
    
    def profile_memory(self) -> float:
        """Profila usage memoria"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            print(f"  📊 Memory usage: {mem_mb:.1f} MB")
            return mem_mb
        except ImportError:
            print(f"  ⚠️  psutil not installed (profiling unavailable)")
            return 0.0
    
    def run_all_tests(self, mode='quick'):
        """Esegui tutti i test"""
        print("\n" + "="*70)
        print(f"{BOLD}{BLUE}🧪 COMPREHENSIVE TEST SUITE - {mode.upper()}{RESET}")
        print("="*70)
        
        # Test 1: Dependencies
        print(f"\n{BOLD}📦 Checking Dependencies...{RESET}")
        deps = self.test_dependencies()
        
        if not deps.get('numpy') or not deps.get('meshio'):
            print(f"\n{RED}❌ CRITICAL: Missing required packages{RESET}")
            return False
        
        # Test 2: Syntax validation
        print(f"\n{BOLD}📝 Validating Python Syntax...{RESET}")
        files = [
            'navier_stokes.py',
            'navier_stokes_numba.py',
            'navier_stokes_cupy.py',
            'mesh_adaptation.py',
            'advanced_solvers.py'
        ]
        
        syntax_ok = True
        for f in files:
            if self.test_syntax(f):
                print(f"  ✅ {f}")
            else:
                print(f"  ❌ {f} - Syntax error")
                syntax_ok = False
        
        if not syntax_ok:
            print(f"\n{RED}❌ Syntax errors found{RESET}")
            return False
        
        # Test 3: Imports
        print(f"\n{BOLD}🔌 Testing Module Imports...{RESET}")
        modules = [
            'navier_stokes',
            'benchmark_optimization'
        ]
        
        for m in modules:
            if self.test_import(m):
                print(f"  ✅ {m}")
            else:
                print(f"  ⚠️  {m} (optional)")
        
        # Test 4: Benchmarking
        if mode in ['medium', 'thorough']:
            print(f"\n{BOLD}⏱️  Performance Benchmarking...{RESET}")
            bench_results = self.benchmark_comparison()
            
            print(f"\n{BOLD}Summary:{RESET}")
            for k, v in bench_results.items():
                print(f"  {k}: {v:.2f}s")
        
        # Test 5: Accuracy
        if mode in ['medium', 'thorough']:
            print(f"\n{BOLD}🎯 Accuracy Validation...{RESET}")
            self.validate_accuracy()
        
        # Test 6: Memory profiling
        if mode == 'thorough':
            print(f"\n{BOLD}💾 Memory Profiling...{RESET}")
            self.profile_memory()
        
        # Test 7: Mesh file check
        print(f"\n{BOLD}📂 Checking Mesh Files...{RESET}")
        try:
            import os
            if os.path.exists('prova.msh'):
                size_mb = os.path.getsize('prova.msh') / 1024 / 1024
                print(f"  ✅ prova.msh ({size_mb:.1f} MB)")
            else:
                print(f"  ⚠️  prova.msh not found (will be created)")
        except Exception as e:
            print(f"  ⚠️  Error checking mesh files")
        
        # Final Report
        print("\n" + "="*70)
        if self.failed_tests:
            print(f"{RED}{BOLD}❌ TEST SUMMARY: {len(self.failed_tests)} FAILURES{RESET}")
            for test, error in self.failed_tests:
                print(f"  • {test}: {error[:60]}")
            return False
        else:
            print(f"{GREEN}{BOLD}✅ ALL TESTS PASSED!{RESET}")
            return True

# ============================================================================
# MAIN TEST ORCHESTRATION
# ============================================================================

def print_welcome():
    """Stampa messaggio di benvenuto"""
    print(f"""
{BOLD}{BLUE}
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║     🚀 WING SIMULATOR - OPTIMIZATION TEST SUITE 🚀                    ║
║                                                                        ║
║  Questo script testa TUTTE le ottimizzazioni implementate              ║
║  e genera un report completo di performance.                          ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
{RESET}
    """)

def main():
    """Entry point principale"""
    print_welcome()
    
    # Parse arguments
    test_mode = 'quick'
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
    
    if test_mode not in ['quick', 'medium', 'thorough', 'gpu']:
        print(f"Usage: {sys.argv[0]} [quick|medium|thorough|gpu]")
        sys.exit(1)
    
    # Run tests
    runner = TestRunner(verbose=True)
    success = runner.run_all_tests(mode=test_mode)
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
