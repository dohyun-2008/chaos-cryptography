
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.config import (
    TPM_PARAMS, CHAOS_PARAMS, NUM_SEEDS, SEED_BASE, MAX_SYNC_STEPS,
    LEARNING_RULE, IMAGE_PATH, CHAOS_MODES, CORRELATION_SAMPLES,
    FIGURES_DIR, LOGS_DIR, CSV_DIR, IMAGES_DIR, PLOT_FORMATS, PLOT_DPI,
    DEBUG_CHAOS_VALUES, DEBUG_SAMPLE_SIZE,
    LOG_LEVEL, LOG_INDIVIDUAL_EXPERIMENTS,
    QUICK_TEST_MODE, QUICK_TEST_SEEDS, QUICK_TEST_MODES, QUICK_TEST_MAX_SYNC_STEPS,
    PROJECT_ROOT
)

from chaos.logistic import generate_logistic
from chaos.tent import generate_tent
from chaos.combined import generate_interleaved, generate_cascade
from tpm import TPM, synchronize
from crypto.image_encrypt import (
    load_image, generate_keystream, encrypt_image, decrypt_image, verify_decryption,
    save_image
)
from metrics.entropy import calculate_entropy
from metrics.correlation import calculate_correlation
from metrics.npcr import calculate_npcr
from metrics.sensitivity import calculate_sensitivity


def setup_logging(log_file: Path):
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)


def process_single_image(
    image_path: Path,
    timestamp: str,
    effective_num_seeds: int,
    effective_chaos_modes: List[str],
    effective_max_sync_steps: int,
    image_idx: int,
    total_images: int
) -> List[Dict]:
    log_file = LOGS_DIR / f"experiment_{timestamp}.log"
    logger = setup_logging(log_file)
    
    import experiments.run_experiments as exp_module
    exp_module._effective_num_seeds = effective_num_seeds
    exp_module._effective_chaos_modes = effective_chaos_modes
    exp_module._effective_max_sync_steps = effective_max_sync_steps
    
    image_name = image_path.name
    logger.info(f"[IMAGE {image_idx+1}/{total_images}] {'='*70}")
    logger.info(f"[IMAGE {image_idx+1}/{total_images}] Processing: {image_name}")
    logger.info(f"[IMAGE {image_idx+1}/{total_images}] Full path: {image_path}")
    
    try:
        image_start_time = time.time()
        logger.info(f"[IMAGE {image_idx+1}/{total_images}] Loading image...")
        image, image_mode = load_image(str(image_path))
        load_time = time.time() - image_start_time
        logger.info(f"[IMAGE {image_idx+1}/{total_images}] [OK] Image loaded in {load_time:.3f}s")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}]   Shape: {image.shape}, Mode: {image_mode}")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}]   Total pixels: {np.prod(image.shape):,}")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}]   Memory size: {image.nbytes / (1024*1024):.2f} MB")
        
        all_results = []
        total_experiments = len(effective_chaos_modes) * effective_num_seeds
        experiment_count = 0
        
        for chaos_mode_idx, chaos_mode in enumerate(effective_chaos_modes):
            mode_start_time = time.time()
            logger.info(f"[IMAGE {image_idx+1}/{total_images}] {'-'*70}")
            logger.info(f"[IMAGE {image_idx+1}/{total_images}] [MODE {chaos_mode_idx+1}/{len(effective_chaos_modes)}] Processing {chaos_mode} mode...")
            logger.info(f"[IMAGE {image_idx+1}/{total_images}] [MODE {chaos_mode_idx+1}/{len(effective_chaos_modes)}] Seeds to process: {effective_num_seeds}")
            
            for seed_idx in range(effective_num_seeds):
                seed = SEED_BASE + seed_idx
                experiment_count += 1
                exp_start_time = time.time()
                
                logger.info(f"[IMAGE {image_idx+1}/{total_images}] [EXP {experiment_count}/{total_experiments}] "
                          f"Mode: {chaos_mode} | Seed: {seed} | Progress: {experiment_count}/{total_experiments} "
                          f"({100*experiment_count/total_experiments:.1f}%)")
                
                result = run_single_experiment(
                    chaos_mode, seed, image, image_mode, logger,
                    save_images=(seed == SEED_BASE),
                    timestamp=timestamp,
                    image_name=image_name,
                    image_idx=image_idx
                )
                
                exp_time = time.time() - exp_start_time
                result['image_path'] = str(image_path)
                result['image_name'] = image_name
                result['image_idx'] = image_idx
                all_results.append(result)
                
                logger.info(f"[IMAGE {image_idx+1}/{total_images}] [EXP {experiment_count}/{total_experiments}] "
                          f"[OK] Completed in {exp_time:.2f}s | Entropy: {result.get('entropy', 'N/A'):.6f} | "
                          f"NPCR: {result.get('npcr', 'N/A'):.2f}% | Sync: {result.get('tpm_sync_steps', 'N/A')} steps")
            
            mode_time = time.time() - mode_start_time
            logger.info(f"[IMAGE {image_idx+1}/{total_images}] [MODE {chaos_mode_idx+1}/{len(effective_chaos_modes)}] "
                      f"[OK] {chaos_mode} mode completed in {mode_time:.2f}s ({mode_time/60:.2f} min)")
        
        total_time = time.time() - image_start_time
        logger.info(f"[IMAGE {image_idx+1}/{total_images}] {'='*70}")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}] [OK] All experiments completed for {image_name}")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}]   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}]   Average per experiment: {total_time/total_experiments:.2f}s")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}]   Total results: {len(all_results)}")
        logger.info(f"[IMAGE {image_idx+1}/{total_images}] {'='*70}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"[IMAGE {image_idx+1}/{total_images}] [ERROR] ERROR processing {image_name}: {e}", exc_info=True)
        return [{
            'image_path': str(image_path),
            'image_name': image_name,
            'image_idx': image_idx,
            'error': str(e)
        }]


def run_single_experiment(
    chaos_mode: str,
    seed: int,
    image: np.ndarray,
    image_mode: str,
    logger: logging.Logger,
    save_images: bool = False,
    timestamp: str = None,
    image_name: str = None,
    image_idx: int = None
) -> Dict:
    if LOG_INDIVIDUAL_EXPERIMENTS and not DEBUG_CHAOS_VALUES:
        logger.info(f"Running experiment: {chaos_mode}, seed={seed}")

    np.random.seed(seed)
    chaos_params = {
        'logistic_r': CHAOS_PARAMS['logistic_r'],
        'tent_r': CHAOS_PARAMS['tent_r'],
    }

    np.random.seed(seed)
    x0 = CHAOS_PARAMS['default_x0'] + np.random.uniform(-0.1, 0.1)
    if DEBUG_CHAOS_VALUES:
        logger.info(f"[INIT] {chaos_mode} | seed={seed} | x0={x0:.6f} | TPM_seeds=({seed}, {seed+1000})")

    # TPM seeds
    tpm_seed_a = seed
    tpm_seed_b = seed + 1000

    start_time = time.time()
    
    # Get effective max_sync_steps (from quick test mode if enabled)
    import experiments.run_experiments as exp_module
    effective_max_sync = getattr(exp_module, '_effective_max_sync_steps', MAX_SYNC_STEPS)
    
    # Enhanced logging for experiment start
    image_info = f" | Image: {image_name}" if image_name else ""
    logger.info(f"[EXP] Starting experiment: {chaos_mode} | seed={seed}{image_info}")
    logger.info(f"[EXP] Parameters: x0={x0:.6f}, TPM_seeds=({tpm_seed_a}, {tpm_seed_b}), max_sync={effective_max_sync}")

    try:
        
        # Generate keystream
        logger.info(f"[EXP] [STEP 1/5] Generating keystream...")
        keystream_start = time.time()
        keystream, sync_info = generate_keystream(
            image_shape=image.shape,
            chaos_mode=chaos_mode,
            x0=x0,
            chaos_params=chaos_params,
            tpm_params=TPM_PARAMS,
            tpm_seed_a=tpm_seed_a,
            tpm_seed_b=tpm_seed_b,
            learning_rule=LEARNING_RULE,
            max_sync_steps=effective_max_sync
        )
        keystream_time = time.time() - keystream_start
        logger.info(f"[EXP] [STEP 1/5] [OK] Keystream generated in {keystream_time:.3f}s")
        logger.info(f"[EXP] [STEP 1/5]   Keystream shape: {keystream.shape}, size: {keystream.nbytes / (1024*1024):.2f} MB")
        logger.info(f"[EXP] [STEP 1/5]   TPM sync: {'SUCCESS' if sync_info['synchronized'] else 'FAILED'} in {sync_info['steps']} steps")

        # Validate keystream randomness
        logger.info(f"[EXP] [STEP 2/5] Validating keystream...")
        keystream_entropy = calculate_entropy(keystream)
        logger.info(f"[EXP] [STEP 2/5] [OK] Keystream entropy: {keystream_entropy:.6f}")
        if keystream_entropy < 7.0:
            logger.warning(f"[EXP] [STEP 2/5] [WARN] Low keystream entropy ({keystream_entropy:.3f}) for {chaos_mode}, seed={seed}")

        # Encrypt image
        logger.info(f"[EXP] [STEP 3/5] Encrypting image...")
        encrypt_start = time.time()
        encrypted = encrypt_image(image, keystream)
        encrypt_time = time.time() - encrypt_start
        logger.info(f"[EXP] [STEP 3/5] [OK] Image encrypted in {encrypt_time:.3f}s")

        # Decrypt to verify
        logger.info(f"[EXP] [STEP 4/5] Decrypting and verifying...")
        decrypt_start = time.time()
        decrypted = decrypt_image(encrypted, keystream)
        is_correct = verify_decryption(image, decrypted)
        decrypt_time = time.time() - decrypt_start
        logger.info(f"[EXP] [STEP 4/5] [OK] Decryption {'SUCCESS' if is_correct else 'FAILED'} in {decrypt_time:.3f}s")

        if not is_correct:
            logger.warning(f"[EXP] [STEP 4/5] [WARN] Decryption verification failed for {chaos_mode}, seed={seed}")

        # Save images if requested (only for first seed of each mode to avoid too many files)
        if save_images and seed == SEED_BASE and timestamp:
            try:
                if image_name:
                    # Use image name in path for multiple images
                    safe_image_name = Path(image_name).stem
                    mode_dir = IMAGES_DIR / timestamp / safe_image_name / chaos_mode
                else:
                    mode_dir = IMAGES_DIR / timestamp / chaos_mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                
                # Save original, encrypted, and decrypted images
                save_image(image, str(mode_dir / "original.png"), image_mode)
                save_image(encrypted, str(mode_dir / "encrypted.png"), image_mode)
                save_image(decrypted, str(mode_dir / "decrypted.png"), image_mode)
                logger.info(f"Saved images to {mode_dir}")
            except Exception as e:
                logger.warning(f"Failed to save images: {e}")

        # Calculate metrics
        logger.info(f"[EXP] [STEP 5/5] Calculating security metrics...")
        metrics_start = time.time()
        
        logger.info(f"[EXP] [STEP 5/5]   Computing entropy...")
        entropy = calculate_entropy(encrypted)
        logger.info(f"[EXP] [STEP 5/5]   [OK] Entropy: {entropy:.6f}")
        
        logger.info(f"[EXP] [STEP 5/5]   Computing correlation (samples: {CORRELATION_SAMPLES})...")
        correlation = calculate_correlation(encrypted, num_samples=CORRELATION_SAMPLES)
        logger.info(f"[EXP] [STEP 5/5]   [OK] Correlation - H: {correlation['horizontal']:.6f}, "
                   f"V: {correlation['vertical']:.6f}, D: {correlation['diagonal']:.6f}")

        # For NPCR and sensitivity, we need a second encrypted image with slightly different seed
        logger.info(f"[EXP] [STEP 5/5]   Generating alternate encryption for NPCR/sensitivity...")
        np.random.seed(seed + 1)
        x0_alt = CHAOS_PARAMS['default_x0'] + np.random.uniform(-0.1, 0.1)
        keystream_alt, sync_info_alt = generate_keystream(
            image_shape=image.shape,
            chaos_mode=chaos_mode,
            x0=x0_alt,
            chaos_params=chaos_params,
            tpm_params=TPM_PARAMS,
            tpm_seed_a=tpm_seed_a + 1,
            tpm_seed_b=tpm_seed_b + 1,
            learning_rule=LEARNING_RULE,
            max_sync_steps=effective_max_sync
        )
        encrypted_alt = encrypt_image(image, keystream_alt)
        logger.info(f"[EXP] [STEP 5/5]   [OK] Alternate encryption generated (sync: {sync_info_alt['steps']} steps)")

        logger.info(f"[EXP] [STEP 5/5]   Computing NPCR...")
        npcr = calculate_npcr(encrypted, encrypted_alt)
        logger.info(f"[EXP] [STEP 5/5]   [OK] NPCR: {npcr:.4f}%")
        
        logger.info(f"[EXP] [STEP 5/5]   Computing sensitivity...")
        sensitivity = calculate_sensitivity(encrypted, encrypted_alt)
        logger.info(f"[EXP] [STEP 5/5]   [OK] Sensitivity - Hamming: {sensitivity['hamming_distance']}, "
                   f"Pixel change: {sensitivity['pixel_change_rate']:.4f}%")
        
        metrics_time = time.time() - metrics_start
        logger.info(f"[EXP] [STEP 5/5] [OK] All metrics calculated in {metrics_time:.3f}s")

        elapsed_time = time.time() - start_time

        result = {
            'chaos_mode': chaos_mode,
            'seed': seed,
            'entropy': entropy,
            'correlation_horizontal': correlation['horizontal'],
            'correlation_vertical': correlation['vertical'],
            'correlation_diagonal': correlation['diagonal'],
            'npcr': npcr,
            'sensitivity_hamming': sensitivity['hamming_distance'],
            'sensitivity_pixel_change_rate': sensitivity['pixel_change_rate'],
            'decryption_correct': is_correct,
            'elapsed_time': elapsed_time,
            'tpm_sync_steps': sync_info['steps'],
            'tpm_synchronized': sync_info['synchronized'],
            'keystream_entropy': keystream_entropy,
        }
        
        # Quality checks
        if abs(correlation['horizontal']) > 0.1:
            logger.warning(f"High correlation ({correlation['horizontal']:.4f}) for {chaos_mode}, seed={seed} - encryption may be weak")
        if npcr < 99.0:
            logger.warning(f"Low NPCR ({npcr:.2f}%) for {chaos_mode}, seed={seed} - sensitivity may be insufficient")
        if not sync_info['synchronized']:
            logger.warning(f"TPM synchronization failed for {chaos_mode}, seed={seed} after {sync_info['steps']} steps")

        if LOG_INDIVIDUAL_EXPERIMENTS:
            image_info = f" | Image: {image_name}" if image_name else ""
            logger.info(f"Completed {chaos_mode}, seed={seed}{image_info} in {elapsed_time:.2f}s")
        return result

    except Exception as e:
        image_info = f" | Image: {image_name}" if image_name else ""
        logger.error(f"Error in experiment {chaos_mode}, seed={seed}{image_info}: {e}", exc_info=True)
        return {
            'chaos_mode': chaos_mode,
            'seed': seed,
            'image_path': str(image_name) if image_name else None,
            'image_name': image_name,
            'image_idx': image_idx,
            'error': str(e)
        }


def save_results_to_csv(results: List[Dict], output_file: Path):
    """Save experiment results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")


def analyze_result_quality(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Analyze result quality and identify potential issues."""
    quality_report = []
    
    for mode in df['chaos_mode'].unique():
        mode_df = df[df['chaos_mode'] == mode]
        mode_label = mode
        
        report = {
            'chaos_mode': mode,
            'num_experiments': len(mode_df),
            'avg_entropy': mode_df['entropy'].mean(),
            'avg_correlation': abs(mode_df['correlation_horizontal'].mean()),
            'avg_npcr': mode_df['npcr'].mean(),
            'sync_success_rate': mode_df.get('tpm_synchronized', pd.Series([True] * len(mode_df))).mean() * 100,
            'avg_sync_steps': mode_df.get('tpm_sync_steps', pd.Series([0] * len(mode_df))).mean(),
        }
        
        # Quality flags
        issues = []
        if report['avg_correlation'] > 0.1:
            issues.append('HIGH_CORRELATION')
        if report['avg_npcr'] < 99.0:
            issues.append('LOW_NPCR')
        if report['avg_entropy'] < 7.9:
            issues.append('LOW_ENTROPY')
        if report['sync_success_rate'] < 100:
            issues.append('SYNC_FAILURES')
        
        report['quality_issues'] = '; '.join(issues) if issues else 'NONE'
        report['quality_score'] = 'GOOD' if not issues else 'POOR'
        
        quality_report.append(report)
        
        # Log findings
        if issues:
            logger.warning(f"{mode_label}: Quality issues detected - {', '.join(issues)}")
            logger.warning(f"  Correlation: {report['avg_correlation']:.4f} (should be <0.1)")
            logger.warning(f"  NPCR: {report['avg_npcr']:.2f}% (should be >99%)")
            logger.warning(f"  Entropy: {report['avg_entropy']:.4f} (should be >7.9)")
            if report['sync_success_rate'] < 100:
                logger.warning(f"  Sync success: {report['sync_success_rate']:.1f}%")
        else:
            logger.info(f"{mode_label}: All quality metrics acceptable")
    
    return pd.DataFrame(quality_report)


def generate_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive statistics table for all metrics."""
    metrics = [
        'entropy', 'correlation_horizontal', 'correlation_vertical', 
        'correlation_diagonal', 'npcr', 'sensitivity_hamming', 
        'sensitivity_pixel_change_rate', 'elapsed_time'
    ]
    
    stats_list = []
    
    for mode in df['chaos_mode'].unique():
        mode_df = df[df['chaos_mode'] == mode]
        stats = {'chaos_mode': mode}
        
        for metric in metrics:
            if metric in mode_df.columns:
                values = mode_df[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_mean'] = values.mean()
                    stats[f'{metric}_std'] = values.std()
                    stats[f'{metric}_min'] = values.min()
                    stats[f'{metric}_max'] = values.max()
                    stats[f'{metric}_median'] = values.median()
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def create_plots(results: List[Dict], logger: logging.Logger):
    """Create and save plots for all metrics with enhanced visualizations."""
    df = pd.DataFrame(results)

    # Filter out rows with errors
    df = df[~df.get('error', pd.Series([False] * len(df))).astype(bool)]

    if len(df) == 0:
        logger.warning("No valid results to plot")
        return
    
    # Removed multiple images support - always single image now

    # Set style for publication quality
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            plt.style.use('default')
    
    # Use a cleaner style for publication
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })
    
    fig_size = (10, 6)

    # 1. Entropy comparison (enhanced with violin plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    df.boxplot(column='entropy', by='chaos_mode', ax=ax1)
    ax1.set_title('Shannon Entropy by Chaos Mode (Box Plot)', fontweight='bold')
    ax1.set_xlabel('Chaos Mode')
    ax1.set_ylabel('Entropy')
    ax1.axhline(y=8.0, color='r', linestyle='--', alpha=0.5, label='Maximum (8.0)')
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Violin plot for better distribution visualization
    modes = df['chaos_mode'].unique()
    data_by_mode = [df[df['chaos_mode'] == mode]['entropy'].values for mode in modes]
    parts = ax2.violinplot(data_by_mode, positions=range(len(modes)), showmeans=True, showmedians=True)
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(modes, rotation=45, ha='right')
    ax2.set_title('Shannon Entropy Distribution (Violin Plot)', fontweight='bold')
    ax2.set_xlabel('Chaos Mode')
    ax2.set_ylabel('Entropy')
    ax2.axhline(y=8.0, color='r', linestyle='--', alpha=0.5, label='Maximum (8.0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    for fmt in PLOT_FORMATS:
        plt.savefig(FIGURES_DIR / f'entropy_comparison.{fmt}', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 2. Correlation comparison (all directions)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    directions = [
        ('correlation_horizontal', 'Horizontal'),
        ('correlation_vertical', 'Vertical'),
        ('correlation_diagonal', 'Diagonal')
    ]
    
    for idx, (col, label) in enumerate(directions):
        ax = axes[idx]
        df.boxplot(column=col, by='chaos_mode', ax=ax)
        ax.set_title(f'{label} Correlation by Chaos Mode', fontweight='bold')
        ax.set_xlabel('Chaos Mode')
        ax.set_ylabel('Correlation Coefficient')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Ideal (0.0)')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')
    
    plt.tight_layout()
    for fmt in PLOT_FORMATS:
        plt.savefig(FIGURES_DIR / f'correlation_all_directions.{fmt}', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 3. NPCR comparison (enhanced)
    fig, ax = plt.subplots(figsize=fig_size)
    df.boxplot(column='npcr', by='chaos_mode', ax=ax)
    ax.set_title('NPCR (Number of Pixels Change Rate) by Chaos Mode', fontweight='bold')
    ax.set_xlabel('Chaos Mode')
    ax.set_ylabel('NPCR (%)')
    ax.axhline(y=99.6, color='g', linestyle='--', alpha=0.5, label='Excellent (>99.6%)')
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.suptitle('')
    for fmt in PLOT_FORMATS:
        plt.savefig(FIGURES_DIR / f'npcr_comparison.{fmt}', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 4. Sensitivity comparison (both metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pixel change rate
    df.boxplot(column='sensitivity_pixel_change_rate', by='chaos_mode', ax=ax1)
    ax1.set_title('Sensitivity: Pixel Change Rate', fontweight='bold')
    ax1.set_xlabel('Chaos Mode')
    ax1.set_ylabel('Pixel Change Rate (%)')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.suptitle('')
    
    # Hamming distance
    df.boxplot(column='sensitivity_hamming', by='chaos_mode', ax=ax2)
    ax2.set_title('Sensitivity: Hamming Distance', fontweight='bold')
    ax2.set_xlabel('Chaos Mode')
    ax2.set_ylabel('Hamming Distance (bits)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.suptitle('')
    
    plt.tight_layout()
    for fmt in PLOT_FORMATS:
        plt.savefig(FIGURES_DIR / f'sensitivity_comparison.{fmt}', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 5. Combined comparison plot (enhanced)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('entropy', 'Shannon Entropy', 'Higher is better'),
        ('correlation_horizontal', 'Horizontal Correlation', 'Closer to 0 is better'),
        ('npcr', 'NPCR (%)', 'Higher is better'),
        ('sensitivity_pixel_change_rate', 'Pixel Change Rate (%)', 'Higher is better')
    ]
    
    for idx, (metric, title, note) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df.boxplot(column=metric, by='chaos_mode', ax=ax)
        ax.set_title(f'{title}\n({note})', fontweight='bold')
        ax.set_xlabel('Chaos Mode', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Security Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    for fmt in PLOT_FORMATS:
        plt.savefig(FIGURES_DIR / f'combined_metrics.{fmt}', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 6. Bar chart with error bars for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    key_metrics = [
        ('entropy', 'Shannon Entropy', axes[0, 0]),
        ('npcr', 'NPCR (%)', axes[0, 1]),
        ('correlation_horizontal', 'Horizontal Correlation', axes[1, 0]),
        ('sensitivity_pixel_change_rate', 'Pixel Change Rate (%)', axes[1, 1])
    ]
    
    for metric, title, ax in key_metrics:
        means = df.groupby('chaos_mode')[metric].mean()
        stds = df.groupby('chaos_mode')[metric].std()
        modes = means.index
        
        bars = ax.bar(range(len(modes)), means.values, yerr=stds.values, 
                     capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=45, ha='right')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means.values, stds.values)):
            ax.text(i, mean + std + (max(means.values) - min(means.values)) * 0.02,
                   f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Mean Values with Standard Deviation', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    for fmt in PLOT_FORMATS:
        plt.savefig(FIGURES_DIR / f'bar_chart_with_errors.{fmt}', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    logger.info("Generated all plots")


def create_image_comparison_plots(timestamp: str, logger: logging.Logger):
    """
    Create comparison plots showing original, encrypted, and decrypted images
    for each chaos mode and image.
    
    Parameters
    ----------
    timestamp : str
        Timestamp string for finding saved images.
    logger : logging.Logger
        Logger instance.
    """
    try:
        from PIL import Image
        
        images_dir = IMAGES_DIR / timestamp
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return
        
        # Find all directories - could be image_name/chaos_mode or just chaos_mode
        all_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
        
        # Check if we have the new structure (image_name/chaos_mode) or old structure (chaos_mode)
        # New structure: each dir contains chaos_mode subdirs
        # Old structure: each dir is a chaos_mode
        image_dirs = []
        chaos_modes = []
        
        # Try to detect structure by checking if first dir contains chaos mode subdirs
        if all_dirs:
            first_dir = all_dirs[0]
            subdirs = [d.name for d in first_dir.iterdir() if d.is_dir()]
            # Check if subdirs match known chaos modes
            known_modes = set(CHAOS_MODES)
            if subdirs and any(mode in known_modes for mode in subdirs):
                # New structure: image_name/chaos_mode
                logger.info(f"Detected new directory structure (image_name/chaos_mode)")
                for img_dir in all_dirs:
                    img_modes = [d.name for d in img_dir.iterdir() if d.is_dir() and d.name in known_modes]
                    for mode in img_modes:
                        if (img_dir.name, mode) not in [(id, m) for id, m in image_dirs]:
                            image_dirs.append((img_dir.name, mode))
                            if mode not in chaos_modes:
                                chaos_modes.append(mode)
            else:
                # Old structure: chaos_mode
                logger.info(f"Detected old directory structure (chaos_mode)")
                chaos_modes = [d.name for d in all_dirs if d.name in known_modes]
                image_dirs = [(None, mode) for mode in chaos_modes]
        
        if not chaos_modes:
            logger.warning(f"No chaos mode directories found in {images_dir}")
            return
        
        # Create comparison plot for each mode (and image if multiple images)
        for img_name, mode in image_dirs:
            if img_name:
                mode_dir = images_dir / img_name / mode
                plot_title_suffix = f" - {Path(img_name).stem}"
            else:
                mode_dir = images_dir / mode
                plot_title_suffix = ""
            
            # Check if all required images exist
            original_path = mode_dir / "original.png"
            encrypted_path = mode_dir / "encrypted.png"
            decrypted_path = mode_dir / "decrypted.png"
            
            if not all([original_path.exists(), encrypted_path.exists(), decrypted_path.exists()]):
                logger.warning(f"Missing images for {mode}, skipping comparison plot")
                continue
            
            # Load images
            try:
                original = Image.open(original_path)
                encrypted = Image.open(encrypted_path)
                decrypted = Image.open(decrypted_path)
            except Exception as e:
                logger.error(f"Failed to load images for {mode}: {e}")
                continue
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original, cmap='gray' if original.mode == 'L' else None)
            axes[0].set_title('Original Image', fontsize=18, fontweight='bold')
            axes[0].axis('off')
            
            # Encrypted image
            axes[1].imshow(encrypted, cmap='gray' if encrypted.mode == 'L' else None)
            axes[1].set_title('Encrypted Image', fontsize=18, fontweight='bold')
            axes[1].axis('off')
            
            # Decrypted image
            axes[2].imshow(decrypted, cmap='gray' if decrypted.mode == 'L' else None)
            axes[2].set_title('Decrypted Image', fontsize=18, fontweight='bold')
            axes[2].axis('off')
            
            plt.suptitle(f'Image Encryption Comparison - {mode.capitalize()} Mode{plot_title_suffix}', 
                        fontsize=22, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save in multiple formats
            for fmt in PLOT_FORMATS:
                if img_name:
                    output_path = FIGURES_DIR / f"image_comparison_{Path(img_name).stem}_{mode}.{fmt}"
                else:
                    output_path = FIGURES_DIR / f"image_comparison_{mode}.{fmt}"
                plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created image comparison plot for {mode}{plot_title_suffix}")
        
        # Create a combined comparison plot with all modes
        num_modes = len(chaos_modes)
        if num_modes > 0:
            fig, axes = plt.subplots(num_modes, 3, figsize=(15, 5 * num_modes))
            
            if num_modes == 1:
                axes = axes.reshape(1, -1)
            
            # For combined plot, use first image if multiple images exist
            row_idx = 0
            for img_name, mode in image_dirs:
                if row_idx >= num_modes:
                    break
                    
                if img_name:
                    mode_dir = images_dir / img_name / mode
                else:
                    mode_dir = images_dir / mode
                
                original_path = mode_dir / "original.png"
                encrypted_path = mode_dir / "encrypted.png"
                decrypted_path = mode_dir / "decrypted.png"
                
                if not all([original_path.exists(), encrypted_path.exists(), decrypted_path.exists()]):
                    continue
                
                try:
                    original = Image.open(original_path)
                    encrypted = Image.open(encrypted_path)
                    decrypted = Image.open(decrypted_path)
                except Exception as e:
                    logger.error(f"Failed to load images for {mode}: {e}")
                    continue
                
                # Original
                axes[row_idx, 0].imshow(original, cmap='gray' if original.mode == 'L' else None)
                if row_idx == 0:
                    axes[row_idx, 0].set_title('Original', fontsize=16, fontweight='bold')
                axes[row_idx, 0].set_ylabel(mode.capitalize(), fontsize=16, fontweight='bold')
                axes[row_idx, 0].axis('off')
                
                # Encrypted
                axes[row_idx, 1].imshow(encrypted, cmap='gray' if encrypted.mode == 'L' else None)
                if row_idx == 0:
                    axes[row_idx, 1].set_title('Encrypted', fontsize=16, fontweight='bold')
                axes[row_idx, 1].axis('off')
                
                # Decrypted
                axes[row_idx, 2].imshow(decrypted, cmap='gray' if decrypted.mode == 'L' else None)
                if row_idx == 0:
                    axes[row_idx, 2].set_title('Decrypted', fontsize=16, fontweight='bold')
                axes[row_idx, 2].axis('off')
                
                row_idx += 1
            
            plt.suptitle('Image Encryption Comparison - All Chaos Modes', 
                        fontsize=24, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            for fmt in PLOT_FORMATS:
                output_path = FIGURES_DIR / f"image_comparison_all_modes.{fmt}"
                plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info("Created combined image comparison plot for all modes")
            
    except ImportError:
        logger.warning("PIL (Pillow) not available, skipping image comparison plots")
    except Exception as e:
        logger.error(f"Error creating image comparison plots: {e}", exc_info=True)


def generate_text_report(df: pd.DataFrame, stats_df: pd.DataFrame, timestamp: str, logger: logging.Logger):
    """Generate a comprehensive text report for the research paper."""
    report_file = LOGS_DIR / f"experiment_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHAOS-TPM HYBRID ENCRYPTION FRAMEWORK - EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Experiments: {len(df)}\n")
        f.write(f"Chaos Modes Tested: {', '.join(df['chaos_mode'].unique())}\n")
        f.write(f"Number of Seeds per Mode: {len(df) // len(df['chaos_mode'].unique())}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENTAL PARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"TPM Parameters:\n")
        for key, value in TPM_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nChaos Parameters:\n")
        for key, value in CHAOS_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nLearning Rule: {LEARNING_RULE}\n")
        f.write(f"Max Synchronization Steps: {MAX_SYNC_STEPS}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS BY CHAOS MODE\n")
        f.write("=" * 80 + "\n\n")
        
        # Key metrics summary
        key_metrics = ['entropy', 'npcr', 'correlation_horizontal', 'sensitivity_pixel_change_rate']
        
        for metric in key_metrics:
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Mode':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in stats_df.iterrows():
                mode = row['chaos_mode']
                mean = row.get(f'{metric}_mean', 'N/A')
                std = row.get(f'{metric}_std', 'N/A')
                min_val = row.get(f'{metric}_min', 'N/A')
                max_val = row.get(f'{metric}_max', 'N/A')
                median = row.get(f'{metric}_median', 'N/A')
                
                if mean != 'N/A':
                    f.write(f"{mode:<15} {mean:<12.6f} {std:<12.6f} {min_val:<12.6f} {max_val:<12.6f} {median:<12.6f}\n")
                else:
                    f.write(f"{mode:<15} {mean:<12} {std:<12} {min_val:<12} {max_val:<12} {median:<12}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        time_stats = df.groupby('chaos_mode')['elapsed_time'].agg(['mean', 'std', 'min', 'max'])
        f.write(f"{'Mode':<15} {'Mean Time (s)':<15} {'Std (s)':<15} {'Min (s)':<15} {'Max (s)':<15}\n")
        f.write("-" * 80 + "\n")
        for mode, row in time_stats.iterrows():
            f.write(f"{mode:<15} {row['mean']:<15.3f} {row['std']:<15.3f} {row['min']:<15.3f} {row['max']:<15.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")
        
        # Find best performing mode for each metric
        for metric in key_metrics:
            if metric == 'correlation_horizontal':
                # Lower is better
                best_mode = stats_df.loc[stats_df[f'{metric}_mean'].idxmin(), 'chaos_mode']
                best_value = stats_df[f'{metric}_mean'].min()
                f.write(f"Best {metric.replace('_', ' ')} (lowest): {best_mode} ({best_value:.6f})\n")
            else:
                # Higher is better
                best_mode = stats_df.loc[stats_df[f'{metric}_mean'].idxmax(), 'chaos_mode']
                best_value = stats_df[f'{metric}_mean'].max()
                f.write(f"Best {metric.replace('_', ' ')} (highest): {best_mode} ({best_value:.6f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DECRYPTION VERIFICATION\n")
        f.write("=" * 80 + "\n\n")
        
        decryption_stats = df.groupby('chaos_mode')['decryption_correct'].agg(['sum', 'count'])
        decryption_stats['success_rate'] = (decryption_stats['sum'] / decryption_stats['count']) * 100
        f.write(f"{'Mode':<15} {'Successful':<15} {'Total':<15} {'Success Rate (%)':<15}\n")
        f.write("-" * 80 + "\n")
        for mode, row in decryption_stats.iterrows():
            f.write(f"{mode:<15} {int(row['sum']):<15} {int(row['count']):<15} {row['success_rate']:<15.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Generated text report: {report_file}")


def generate_latex_tables(df: pd.DataFrame, stats_df: pd.DataFrame, timestamp: str, logger: logging.Logger):
    """Generate LaTeX tables for easy inclusion in research papers."""
    latex_file = LOGS_DIR / f"latex_tables_{timestamp}.tex"
    
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("% LaTeX Tables for Chaos-TPM Encryption Framework\n")
        f.write("% Generated automatically from experiment results\n\n")
        
        # Table 1: Summary Statistics
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics of Security Metrics by Chaos Mode}\n")
        f.write("\\label{tab:summary_stats}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Chaos Mode & Entropy & NPCR (\\%) & Correlation & Sensitivity (\\%)\\\\\n")
        f.write("\\midrule\n")
        
        for _, row in stats_df.iterrows():
            mode = row['chaos_mode']
            entropy = row.get('entropy_mean', 0)
            npcr = row.get('npcr_mean', 0)
            corr = row.get('correlation_horizontal_mean', 0)
            sens = row.get('sensitivity_pixel_change_rate_mean', 0)
            
            f.write(f"{mode.capitalize()} & {entropy:.4f} & {npcr:.2f} & {corr:.4f} & {sens:.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Table 2: Detailed Entropy Statistics
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Detailed Entropy Statistics}\n")
        f.write("\\label{tab:entropy_stats}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Mode & Mean & Std & Min & Max & Median\\\\\n")
        f.write("\\midrule\n")
        
        for _, row in stats_df.iterrows():
            mode = row['chaos_mode']
            mean = row.get('entropy_mean', 0)
            std = row.get('entropy_std', 0)
            min_val = row.get('entropy_min', 0)
            max_val = row.get('entropy_max', 0)
            median = row.get('entropy_median', 0)
            
            f.write(f"{mode.capitalize()} & {mean:.6f} & {std:.6f} & {min_val:.6f} & {max_val:.6f} & {median:.6f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Table 3: Performance Metrics
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Metrics (Execution Time)}\n")
        f.write("\\label{tab:performance}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Chaos Mode & Mean (s) & Std (s) & Min (s) & Max (s)\\\\\n")
        f.write("\\midrule\n")
        
        time_stats = df.groupby('chaos_mode')['elapsed_time'].agg(['mean', 'std', 'min', 'max'])
        for mode, row in time_stats.iterrows():
            f.write(f"{mode.capitalize()} & {row['mean']:.3f} & {row['std']:.3f} & {row['min']:.3f} & {row['max']:.3f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logger.info(f"Generated LaTeX tables: {latex_file}")


def find_image_files(data_dir: Path) -> List[Path]:
    """Find all image files in the data directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    image_files = []
    
    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def main():
    """Main experiment execution function."""
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"experiment_{timestamp}.log"
    logger = setup_logging(log_file)

    # Apply quick test mode settings if enabled
    effective_num_seeds = QUICK_TEST_SEEDS if QUICK_TEST_MODE else NUM_SEEDS
    effective_chaos_modes = QUICK_TEST_MODES if QUICK_TEST_MODE else CHAOS_MODES
    effective_max_sync_steps = QUICK_TEST_MAX_SYNC_STEPS if QUICK_TEST_MODE else MAX_SYNC_STEPS
    
    # Store effective values for worker processes and functions
    # These will be used by process_single_image and run_single_experiment
    import experiments.run_experiments as exp_module
    exp_module._effective_num_seeds = effective_num_seeds
    exp_module._effective_chaos_modes = effective_chaos_modes
    exp_module._effective_max_sync_steps = effective_max_sync_steps

    logger.info("=" * 80)
    logger.info("Starting Chaos-TPM Encryption Experiments")
    if QUICK_TEST_MODE:
        logger.info("*** QUICK TEST MODE ENABLED ***")
        logger.info(f"  - Seeds: {effective_num_seeds} (instead of {NUM_SEEDS})")
        logger.info(f"  - Modes: {effective_chaos_modes} (instead of {CHAOS_MODES})")
        logger.info(f"  - Max sync steps: {effective_max_sync_steps} (instead of {MAX_SYNC_STEPS})")
    logger.info("=" * 80)
    logger.info(f"TPM Parameters: {TPM_PARAMS}")
    logger.info(f"Chaos Parameters: {CHAOS_PARAMS}")
    logger.info(f"Number of seeds: {effective_num_seeds}")
    logger.info(f"Chaos modes: {effective_chaos_modes}")
    logger.info(f"Max sync steps: {effective_max_sync_steps}")
    
    # Find all images in data directory
    data_dir = PROJECT_ROOT / "data"
    image_files = find_image_files(data_dir)
    
    if not image_files:
        logger.warning(f"No image files found in {data_dir}")
        logger.info(f"Looking for images with extensions: .png, .jpg, .jpeg, .bmp, .tiff, .tif, .gif")
        # Fallback to single image path
        if IMAGE_PATH.exists():
            logger.info(f"Falling back to single image: {IMAGE_PATH}")
            image_files = [IMAGE_PATH]
        else:
            logger.error(f"Image not found: {IMAGE_PATH}")
            logger.error("Please place a sample image at data/sample.png or add images to data/ directory")
            return
    else:
        logger.info(f"Found {len(image_files)} image(s) to test")
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"  [{idx}] {img_path.name} ({img_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Determine if we should use parallel processing
    use_parallel = len(image_files) > 1
    max_workers = None  # Auto-detect CPU count
    
    if use_parallel:
        cpu_count = os.cpu_count() or 4
        max_workers = min(cpu_count, len(image_files))
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Using parallel processing ({'process' if use_parallel else 'sequential'}) for {len(image_files)} images")
        logger.info(f"Max workers: {max_workers} (CPU count: {cpu_count})")
        logger.info(f"Estimated total experiments: {len(image_files) * len(effective_chaos_modes) * effective_num_seeds}")
        logger.info("=" * 80)
        logger.info("")
    
    # Process images
    all_results = []
    overall_start_time = time.time()
    
    if use_parallel and len(image_files) > 1:
        # Parallel processing for multiple images
        logger.info(f"[PARALLEL] Starting parallel processing with {max_workers} workers...")
        logger.info(f"[PARALLEL] Processing {len(image_files)} images across {len(effective_chaos_modes)} modes × {effective_num_seeds} seeds each")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all image processing tasks
            future_to_image = {
                executor.submit(
                    process_single_image,
                    image_path,
                    timestamp,
                    effective_num_seeds,
                    effective_chaos_modes,
                    effective_max_sync_steps,
                    idx,
                    len(image_files)
                ): (idx, image_path) for idx, image_path in enumerate(image_files)
            }
            
            completed_count = 0
            total_tasks = len(future_to_image)
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_image):
                image_idx, image_path = future_to_image[future]
                completed_count += 1
                
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"[PARALLEL] [OK] Completed image {image_idx+1}/{total_tasks}: {image_path.name} "
                              f"({completed_count}/{total_tasks} done, {100*completed_count/total_tasks:.1f}%)")
                except Exception as e:
                    logger.error(f"[PARALLEL] [ERROR] Error processing {image_path.name}: {e}", exc_info=True)
                    all_results.append({
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'image_idx': image_idx,
                        'error': str(e)
                    })
        
        logger.info(f"[PARALLEL] All parallel tasks completed")
    else:
        # Sequential processing (single image or parallel disabled)
        logger.info(f"[SEQUENTIAL] Processing images sequentially...")
        for idx, image_path in enumerate(image_files):
            results = process_single_image(
                image_path,
                timestamp,
                effective_num_seeds,
                effective_chaos_modes,
                effective_max_sync_steps,
                idx,
                len(image_files)
            )
            all_results.extend(results)
    
    overall_time = time.time() - overall_start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"All image processing completed in {overall_time:.2f}s ({overall_time/60:.2f} min)")
    logger.info(f"Total results collected: {len(all_results)}")
    
    # Summary statistics by image
    if len(image_files) > 1:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Processing Summary by Image")
        logger.info("=" * 80)
        df_temp = pd.DataFrame(all_results)
        if 'image_name' in df_temp.columns and 'error' not in df_temp.columns:
            for image_name in df_temp['image_name'].unique():
                img_df = df_temp[df_temp['image_name'] == image_name]
                valid_df = img_df[~img_df.get('error', pd.Series([False] * len(img_df))).astype(bool)]
                if len(valid_df) > 0:
                    logger.info(f"  {image_name}:")
                    logger.info(f"    - Experiments: {len(valid_df)}/{len(img_df)} successful")
                    logger.info(f"    - Avg entropy: {valid_df['entropy'].mean():.6f}")
                    logger.info(f"    - Avg NPCR: {valid_df['npcr'].mean():.4f}%")
                    logger.info(f"    - Avg time: {valid_df['elapsed_time'].mean():.2f}s")
                    logger.info(f"    - Total time: {valid_df['elapsed_time'].sum():.2f}s")
        logger.info("=" * 80)
    
    logger.info("")

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving results")
    logger.info("=" * 80)

    csv_file = CSV_DIR / f"results_{timestamp}.csv"
    save_results_to_csv(all_results, csv_file)

    # Create plots
    logger.info("Generating plots...")
    create_plots(all_results, logger)
    
    # Create image comparison plots
    logger.info("Generating image comparison plots...")
    create_image_comparison_plots(timestamp, logger)

    # Generate comprehensive statistics
    df = pd.DataFrame(all_results)
    df_valid = df[~df.get('error', pd.Series([False] * len(df))).astype(bool)]

    if len(df_valid) > 0:
        # Generate detailed statistics table
        stats_df = generate_statistics_table(df_valid)
        stats_file = CSV_DIR / f"statistics_{timestamp}.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved detailed statistics to {stats_file}")

        # Summary statistics for logging
        logger.info("\n" + "=" * 80)
        logger.info("Summary Statistics")
        logger.info("=" * 80)
        summary = df_valid.groupby('chaos_mode').agg({
            'entropy': ['mean', 'std', 'min', 'max'],
            'correlation_horizontal': ['mean', 'std'],
            'npcr': ['mean', 'std', 'min', 'max'],
            'sensitivity_pixel_change_rate': ['mean', 'std'],
            'elapsed_time': ['mean', 'std'],
        })
        logger.info("\n" + str(summary))
        
        # TPM Synchronization Statistics
        if 'tpm_synchronized' in df_valid.columns:
            logger.info("\n" + "=" * 80)
            logger.info("TPM Synchronization Statistics")
            logger.info("=" * 80)
            sync_stats = df_valid.groupby('chaos_mode').agg({
                'tpm_synchronized': ['sum', 'count'],
                'tpm_sync_steps': ['mean', 'std', 'min', 'max']
            })
            sync_stats['sync_rate'] = (sync_stats[('tpm_synchronized', 'sum')] / 
                                      sync_stats[('tpm_synchronized', 'count')]) * 100
            logger.info("\n" + str(sync_stats))
            
            # Warn about synchronization failures
            for mode in df_valid['chaos_mode'].unique():
                mode_df = df_valid[df_valid['chaos_mode'] == mode]
                sync_rate = mode_df['tpm_synchronized'].mean() * 100
                if sync_rate < 100:
                    logger.warning(f"{mode}: Only {sync_rate:.1f}% synchronization success rate")
        
        # Result Quality Assessment
        logger.info("\n" + "=" * 80)
        logger.info("Result Quality Assessment")
        logger.info("=" * 80)
        for mode in df_valid['chaos_mode'].unique():
            mode_df = df_valid[df_valid['chaos_mode'] == mode]
            avg_corr = abs(mode_df['correlation_horizontal'].mean())
            avg_npcr = mode_df['npcr'].mean()
            avg_entropy = mode_df['entropy'].mean()
            
            quality_issues = []
            if avg_corr > 0.1:
                quality_issues.append(f"High correlation ({avg_corr:.4f})")
            if avg_npcr < 99.0:
                quality_issues.append(f"Low NPCR ({avg_npcr:.2f}%)")
            if avg_entropy < 7.9:
                quality_issues.append(f"Low entropy ({avg_entropy:.4f})")
            
            if quality_issues:
                logger.warning(f"{mode}: " + ", ".join(quality_issues))
            else:
                logger.info(f"{mode}: All quality metrics within acceptable ranges")
        
        # Quality analysis
        quality_df = analyze_result_quality(df_valid, logger)
        quality_file = CSV_DIR / f"quality_analysis_{timestamp}.csv"
        quality_df.to_csv(quality_file, index=False)
        logger.info(f"Saved quality analysis to {quality_file}")
        
        # Generate text report
        generate_text_report(df_valid, stats_df, timestamp, logger)
        
        # Generate LaTeX tables
        generate_latex_tables(df_valid, stats_df, timestamp, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Experiments completed successfully!")
    logger.info("=" * 80)
    logger.info("Generated Outputs:")
    logger.info(f"  - Raw Results CSV: {CSV_DIR}/results_{timestamp}.csv")
    logger.info(f"  - Statistics CSV: {CSV_DIR}/statistics_{timestamp}.csv")
    logger.info(f"  - Text Report: {LOGS_DIR}/experiment_report_{timestamp}.txt")
    logger.info(f"  - LaTeX Tables: {LOGS_DIR}/latex_tables_{timestamp}.tex")
    logger.info(f"  - Plots (PNG + PDF): {FIGURES_DIR}/")
    logger.info("    * entropy_comparison.{png,pdf}")
    logger.info("    * correlation_all_directions.{png,pdf}")
    logger.info("    * npcr_comparison.{png,pdf}")
    logger.info("    * sensitivity_comparison.{png,pdf}")
    logger.info("    * combined_metrics.{png,pdf}")
    logger.info("    * bar_chart_with_errors.{png,pdf}")
    logger.info("    * image_comparison_*.{png,pdf}")
    logger.info(f"  - Encrypted Images: {IMAGES_DIR}/{timestamp}/")
    logger.info(f"  - Experiment Log: {LOGS_DIR}/experiment_{timestamp}.log")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

