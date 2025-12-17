"""
Image Encryption Module using Chaos-TPM Hybrid Approach

This module implements XOR-based image encryption using a keystream generated
from chaotic sequences and Tree Parity Machine synchronization.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
from pathlib import Path
import hashlib

try:
    # Try relative imports first (when used as a package)
    from ..tpm import TPM, synchronize
    from ..chaos.logistic import generate_logistic
    from ..chaos.tent import generate_tent
    from ..chaos.combined import generate_interleaved, generate_cascade
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from tpm import TPM, synchronize
    from chaos.logistic import generate_logistic
    from chaos.tent import generate_tent
    from chaos.combined import generate_interleaved, generate_cascade


def generate_keystream(
    image_shape: Tuple[int, ...],
    chaos_mode: str,
    x0: float,
    chaos_params: dict,
    tpm_params: dict,
    tpm_seed_a: int,
    tpm_seed_b: int,
    learning_rule: str = "hebbian",
    max_sync_steps: int = 10000
) -> Tuple[np.ndarray, dict]:
    """
    Generate a keystream for image encryption using chaos-TPM hybrid approach.

    The process:
    1. Generate chaotic sequence based on the specified mode
    2. Initialize two TPMs (A and B)
    3. Synchronize TPMs using the chaotic sequence
    4. Expand TPM keys to match image size

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width) for grayscale or (height, width, channels) for RGB.
    chaos_mode : str
        One of "logistic", "tent", "interleaved", "cascade".
    x0 : float
        Initial condition for chaotic map.
    chaos_params : dict
        Parameters for chaotic map (e.g., {'r': 3.99}).
    tpm_params : dict
        TPM parameters: {'K': int, 'N': int, 'L': int}.
    tpm_seed_a : int
        Random seed for TPM A initialization.
    tpm_seed_b : int
        Random seed for TPM B initialization.
    learning_rule : str, optional
        TPM learning rule. Default: "hebbian".
    max_sync_steps : int, optional
        Maximum synchronization steps. Default: 10000.

    Returns
    -------
    np.ndarray
        Keystream array matching image shape, dtype uint8.
    """
    # Calculate required sequence length
    # Need enough for TPM sync + keystream expansion
    K = tpm_params['K']
    N = tpm_params['N']
    sync_sequence_length = max_sync_steps * K * N
    image_pixels = np.prod(image_shape)
    total_length = sync_sequence_length + image_pixels

    # Generate chaotic sequence
    if chaos_mode == "logistic":
        chaos_seq = generate_logistic(total_length, x0, chaos_params)
    elif chaos_mode == "tent":
        chaos_seq = generate_tent(total_length, x0, chaos_params)
    elif chaos_mode == "interleaved":
        chaos_seq = generate_interleaved(total_length, x0, chaos_params)
    elif chaos_mode == "cascade":
        chaos_seq = generate_cascade(total_length, x0, chaos_params)
    else:
        raise ValueError(f"Unknown chaos mode: {chaos_mode}")

    # Debug: Output chaos sequence values (cleaner format)
    try:
        from experiments.config import DEBUG_CHAOS_VALUES, DEBUG_SAMPLE_SIZE
        if DEBUG_CHAOS_VALUES:
            import logging
            logger = logging.getLogger(__name__)
            
            # Check for invalid values first
            has_inf = np.any(np.isinf(chaos_seq))
            has_nan = np.any(np.isnan(chaos_seq))
            has_negative = np.any(chaos_seq < 0)
            has_over_one = np.any(chaos_seq > 1)
            
            if has_inf or has_nan or has_negative or has_over_one:
                logger.warning(f"[{chaos_mode.upper()}] Invalid values detected: inf={has_inf}, nan={has_nan}, <0={has_negative}, >1={has_over_one}")
                # Clip invalid values
                chaos_seq = np.clip(chaos_seq, 0.0, 1.0)
                chaos_seq = np.nan_to_num(chaos_seq, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Clean statistics (only valid values)
            valid_seq = chaos_seq[np.isfinite(chaos_seq) & (chaos_seq >= 0) & (chaos_seq <= 1)]
            if len(valid_seq) > 0:
                sample_size = min(DEBUG_SAMPLE_SIZE, len(valid_seq))
                sample_values = valid_seq[:min(20, len(valid_seq))]
                
                logger.info(f"[{chaos_mode.upper()}] x0={x0:.6f} | len={len(chaos_seq)} | "
                           f"min={valid_seq.min():.4f} max={valid_seq.max():.4f} mean={valid_seq.mean():.4f} std={valid_seq.std():.4f}")
                logger.info(f"[{chaos_mode.upper()}] First 10: {[f'{v:.4f}' for v in sample_values[:10]]}")
                
                # Distribution balance
                below_half = np.sum(valid_seq < 0.5)
                above_half = np.sum(valid_seq >= 0.5)
                logger.info(f"[{chaos_mode.upper()}] Distribution: <0.5={100*below_half/len(valid_seq):.1f}% | >=0.5={100*above_half/len(valid_seq):.1f}%")
                
                # For combined modes, show separate analysis
                if chaos_mode == "interleaved":
                    logistic_values = valid_seq[::2] if len(valid_seq) > 1 else valid_seq
                    tent_values = valid_seq[1::2] if len(valid_seq) > 1 else valid_seq
                    if len(logistic_values) > 0 and len(tent_values) > 0:
                        logger.info(f"[{chaos_mode.upper()}] Logistic: min={logistic_values.min():.4f} max={logistic_values.max():.4f} mean={logistic_values.mean():.4f}")
                        logger.info(f"[{chaos_mode.upper()}] Tent:     min={tent_values.min():.4f} max={tent_values.max():.4f} mean={tent_values.mean():.4f}")
            else:
                logger.error(f"[{chaos_mode.upper()}] No valid values in sequence!")
    except ImportError:
        pass  # Config not available, skip debug output

    # Split sequence: first part for sync, second for keystream
    sync_seq = chaos_seq[:sync_sequence_length]
    keystream_seq = chaos_seq[sync_sequence_length:sync_sequence_length + image_pixels]

    # Initialize TPMs
    tpm_a = TPM(
        K=tpm_params['K'],
        N=tpm_params['N'],
        L=tpm_params['L'],
        seed=tpm_seed_a
    )
    tpm_b = TPM(
        K=tpm_params['K'],
        N=tpm_params['N'],
        L=tpm_params['L'],
        seed=tpm_seed_b
    )

    # Synchronize TPMs
    try:
        from experiments.config import DEBUG_CHAOS_VALUES
        debug_sync = DEBUG_CHAOS_VALUES
    except ImportError:
        debug_sync = False
    
    steps, key_a, key_b = synchronize(
        tpm_a, tpm_b, sync_seq, max_steps=max_sync_steps, 
        learning_rule=learning_rule, debug=debug_sync
    )
    
    # Check if synchronization succeeded
    is_synced = tpm_a.is_synchronized(tpm_b)
    sync_info = {
        'steps': steps,
        'synchronized': is_synced,
        'max_steps_reached': steps >= max_sync_steps
    }
    
    # Prepare keystream sequence for fallback (needed before the if statement)
    keystream_seq_clamped = np.clip(keystream_seq, 0.0, 1.0)
    
    if not is_synced:
        # If not synchronized, use a better fallback: generate high-quality keystream from chaos
        # Use the entire chaos sequence with proper mixing for better randomness
        import warnings
        
        # Generate a high-quality keystream directly from chaos sequence
        # Method: Use multiple hash rounds with different parts of the chaos sequence
        # This ensures good distribution even if chaos sequence has patterns
        
        # Sample multiple points from chaos sequence for hash input
        num_samples = min(1000, len(keystream_seq_clamped))
        sample_indices = np.linspace(0, len(keystream_seq_clamped) - 1, num_samples, dtype=int)
        chaos_samples = (keystream_seq_clamped[sample_indices] * 255).astype(np.uint8)
        
        # Combine with TPM weights for additional entropy
        weight_bytes = (tpm_a.weights.flatten().astype(np.int32).tobytes() + 
                       tpm_b.weights.flatten().astype(np.int32).tobytes())
        
        # Generate keystream using counter-based hashing for better distribution
        # Use the full chaos sequence for maximum entropy
        chaos_full_bytes = (keystream_seq_clamped * 255).astype(np.uint8).tobytes()
        
        # Create initial hash from chaos sequence and weights
        hash_state = hashlib.sha256(chaos_full_bytes[:min(10000, len(chaos_full_bytes))] + weight_bytes).digest()
        
        # Generate keystream using counter-based approach
        keystream_bytes = bytearray()
        bytes_needed = image_pixels
        
        # Use counter-based hashing: hash(hash_state || counter || chaos_sample) for each block
        counter = 0
        while len(keystream_bytes) < bytes_needed:
            counter_bytes = counter.to_bytes(8, 'big')
            # Sample different parts of chaos sequence for each block
            chaos_idx = counter % len(chaos_full_bytes)
            chaos_sample = chaos_full_bytes[chaos_idx:chaos_idx + 32].ljust(32, b'\x00')
            block_hash = hashlib.sha256(hash_state + counter_bytes + chaos_sample).digest()
            keystream_bytes.extend(block_hash)
            counter += 1
        
        # Convert to numpy array
        keystream_fallback = np.frombuffer(keystream_bytes[:bytes_needed], dtype=np.uint8)
        
        # Additional non-linear mixing with chaos sequence
        chaos_uint8 = (keystream_seq_clamped * 255).astype(np.uint8)
        keystream_final = keystream_fallback.astype(np.uint16).copy()
        chaos_uint16 = chaos_uint8.astype(np.uint16)
        
        # Mix using bit rotation and XOR for better distribution
        # Use uint16 for calculations to avoid overflow
        for i in range(min(len(keystream_final), len(chaos_uint8))):
            rotated = ((chaos_uint8[i] << (i % 8)) | (chaos_uint8[i] >> (8 - (i % 8)))) & 0xFF
            rotated_16 = np.uint16(rotated)
            keystream_final[i] = (keystream_final[i] ^ rotated_16 ^ chaos_uint16[i]) % 256
        
        # Convert back to uint8
        keystream_final = keystream_final.astype(np.uint8)
        
        # Reshape and return directly (skip TPM key expansion)
        keystream = keystream_final.astype(np.uint8).reshape(image_shape)
        return keystream, sync_info
    else:
        # Combine TPM keys with chaos sequence for keystream
        # Convert TPM keys to numpy array
        key_a_array = np.frombuffer(key_a, dtype=np.uint8)
        key_b_array = np.frombuffer(key_b, dtype=np.uint8)

    # Expand keys to match image size
    if len(key_a_array) < image_pixels:
        # Repeat keys
        repeats_a = (image_pixels // len(key_a_array)) + 1
        key_a_expanded = np.tile(key_a_array, repeats_a)[:image_pixels]
    else:
        key_a_expanded = key_a_array[:image_pixels]

    if len(key_b_array) < image_pixels:
        repeats_b = (image_pixels // len(key_b_array)) + 1
        key_b_expanded = np.tile(key_b_array, repeats_b)[:image_pixels]
    else:
        key_b_expanded = key_b_array[:image_pixels]

    # Combine: XOR keys with chaos sequence
    # Use uint16 for intermediate calculation to avoid overflow
    keystream_chaos = (keystream_seq_clamped * 255).astype(np.uint16)
    key_a_expanded_16 = key_a_expanded.astype(np.uint16)
    key_b_expanded_16 = key_b_expanded.astype(np.uint16)
    
    # CRITICAL: Add seed-dependent mixing to ensure keystream sensitivity
    # This ensures different seeds produce different keystreams even if TPMs synchronize similarly
    seed_hash = hashlib.sha256(
        (str(chaos_mode) + str(tpm_seed_a) + str(tpm_seed_b) + str(x0) + 
         str(chaos_params.get('tent_r', 2.0)) + str(chaos_params.get('logistic_r', 3.99))).encode()
    ).digest()
    seed_hash_array = np.frombuffer(seed_hash, dtype=np.uint8)
    # Expand seed hash to match image size
    if len(seed_hash_array) < image_pixels:
        repeats_seed = (image_pixels // len(seed_hash_array)) + 1
        seed_hash_expanded = np.tile(seed_hash_array, repeats_seed)[:image_pixels]
    else:
        seed_hash_expanded = seed_hash_array[:image_pixels]
    seed_hash_expanded_16 = seed_hash_expanded.astype(np.uint16)
    
    # Also include sync steps and sync status for additional entropy
    sync_entropy = np.array([sync_info['steps'] % 256, int(sync_info['synchronized']) * 128], dtype=np.uint8)
    sync_entropy_expanded = np.tile(sync_entropy, (image_pixels // 2 + 1))[:image_pixels]
    sync_entropy_16 = sync_entropy_expanded.astype(np.uint16)
    
    # Perform XOR operations with multiple sources for better mixing
    # Use uint16 for calculations to avoid overflow
    keystream = ((key_a_expanded_16 ^ key_b_expanded_16 ^ keystream_chaos ^ 
                  seed_hash_expanded_16 ^ sync_entropy_16) % 256).astype(np.uint8)

    # Additional non-linear mixing: rotate and XOR with position-dependent values
    # This breaks any remaining patterns and improves entropy
    # Vectorized version for better performance
    indices = np.arange(len(keystream))
    rot_amounts = (indices + keystream.astype(np.uint16)) % 8
    # Vectorized bit rotation (use uint16 to avoid overflow)
    rotated_16 = ((keystream.astype(np.uint16) << rot_amounts) | 
                  (keystream.astype(np.uint16) >> (8 - rot_amounts))) & 0xFF
    # Mix with seed hash and chaos (vectorized, use uint16 for all operations)
    seed_indices = indices % len(seed_hash_expanded)
    keystream_16 = (rotated_16.astype(np.uint16) ^ 
                    seed_hash_expanded[seed_indices].astype(np.uint16) ^ 
                    (keystream_chaos & 0xFF).astype(np.uint16))
    
    # Additional mixing pass: use S-box-like substitution for better distribution
    # Create a pseudo-random substitution table from seed hash
    sbox = np.arange(256, dtype=np.uint8)
    # Shuffle using seed hash as seed (ensure seed is in valid range [0, 2^32-1])
    sbox_seed = int.from_bytes(seed_hash[:8], 'big') % (2**32)
    np.random.seed(sbox_seed)
    np.random.shuffle(sbox)
    np.random.seed(None)  # Reset seed
    
    # Apply substitution
    keystream_16 = sbox[keystream_16.astype(np.uint8)].astype(np.uint16)
    
    # Final mixing: XOR with position-dependent values from chaos sequence
    # Use different parts of chaos sequence for each position
    chaos_indices = (indices * 7 + sync_info['steps']) % len(keystream_seq_clamped)
    chaos_final = (keystream_seq_clamped[chaos_indices] * 255).astype(np.uint16)
    keystream_16 = (keystream_16 ^ chaos_final) & 0xFF
    
    # Apply modulo and cast to uint8 (modulo ensures values are in [0, 255])
    # Use np.clip as additional safety to ensure values are in valid range
    keystream = np.clip(keystream_16 % 256, 0, 255).astype(np.uint8)

    # Reshape to image shape
    keystream = keystream.reshape(image_shape)

    return keystream, sync_info


def encrypt_image(
    image: np.ndarray,
    keystream: np.ndarray
) -> np.ndarray:
    """
    Encrypt an image using XOR operation with keystream.

    Formula: cipher_pixel = (plain_pixel XOR key) mod 256

    Parameters
    ----------
    image : np.ndarray
        Input image array, dtype uint8, shape (H, W) or (H, W, C).
    keystream : np.ndarray
        Keystream array matching image shape, dtype uint8.

    Returns
    -------
    np.ndarray
        Encrypted image array, same shape and dtype as input.
    """
    if image.shape != keystream.shape:
        raise ValueError(f"Image shape {image.shape} must match keystream shape {keystream.shape}")

    encrypted = (image.astype(np.uint16) ^ keystream.astype(np.uint16)) % 256
    return encrypted.astype(np.uint8)


def decrypt_image(
    encrypted: np.ndarray,
    keystream: np.ndarray
) -> np.ndarray:
    """
    Decrypt an image using XOR operation with keystream.

    XOR is symmetric, so decryption is the same as encryption.

    Parameters
    ----------
    encrypted : np.ndarray
        Encrypted image array, dtype uint8.
    keystream : np.ndarray
        Keystream array matching encrypted shape, dtype uint8.

    Returns
    -------
    np.ndarray
        Decrypted image array, same shape and dtype as input.
    """
    return encrypt_image(encrypted, keystream)


def verify_decryption(
    original: np.ndarray,
    decrypted: np.ndarray
) -> bool:
    """
    Verify that decryption correctly recovers the original image.

    Parameters
    ----------
    original : np.ndarray
        Original image array.
    decrypted : np.ndarray
        Decrypted image array.

    Returns
    -------
    bool
        True if images are identical, False otherwise.
    """
    return np.array_equal(original, decrypted)


def load_image(image_path: str) -> Tuple[np.ndarray, str]:
    """
    Load an image from file.

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    image : np.ndarray
        Image array, dtype uint8.
    mode : str
        Image mode ('L' for grayscale, 'RGB' for color).
    """
    img = Image.open(image_path)
    mode = img.mode

    if mode == 'L':
        # Grayscale
        image = np.array(img, dtype=np.uint8)
    elif mode in ('RGB', 'RGBA'):
        # Color
        image = np.array(img, dtype=np.uint8)
        if mode == 'RGBA':
            # Convert RGBA to RGB for simplicity
            image = image[:, :, :3]
    else:
        # Convert to RGB
        img = img.convert('RGB')
        image = np.array(img, dtype=np.uint8)

    return image, mode


def save_image(image: np.ndarray, output_path: str, mode: str = 'L'):
    """
    Save an image array to file.

    Parameters
    ----------
    image : np.ndarray
        Image array, dtype uint8.
    output_path : str
        Output file path.
    mode : str, optional
        Image mode ('L' for grayscale, 'RGB' for color). Default: 'L'.
    """
    if len(image.shape) == 2:
        # Grayscale
        img = Image.fromarray(image, mode='L')
    elif len(image.shape) == 3:
        # Color
        img = Image.fromarray(image, mode='RGB')
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    img.save(output_path)

