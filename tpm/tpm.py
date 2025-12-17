"""Tree Parity Machine (TPM) implementation."""

import numpy as np
from typing import Literal, Optional, Tuple


class TPM:
    """Neural key exchange model."""

    def __init__(
        self,
        K: int,
        N: int,
        L: int,
        seed: Optional[int] = None
    ):
        if K <= 0 or N <= 0 or L <= 0:
            raise ValueError("K, N, and L must be positive integers")

        self.K = K
        self.N = N
        self.L = L

        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.randint(-L, L + 1, size=(K, N), dtype=np.int32)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compute hidden-unit outputs and parity."""
        if x.shape != (self.K, self.N):
            raise ValueError(f"Input must have shape ({self.K}, {self.N}), got {x.shape}")

        weighted_sums = np.sum(self.weights * x, axis=1)
        sigma = np.sign(weighted_sums)
        sigma[sigma == 0] = 1
        tau = int(np.prod(sigma))
        return sigma, tau

    def update(
        self,
        x: np.ndarray,
        tau: int,
        learning_rule: Literal["hebbian", "anti_hebbian", "random_walk"] = "hebbian"
    ):
        """Update weights according to the specified learning rule."""
        sigma, _ = self.predict(x)
        update_mask = (sigma == tau).astype(int)

        if learning_rule == "hebbian":
            delta = x * tau * update_mask[:, np.newaxis]
        elif learning_rule == "anti_hebbian":
            delta = -x * tau * update_mask[:, np.newaxis]
        elif learning_rule == "random_walk":
            delta = x * update_mask[:, np.newaxis]
        else:
            raise ValueError(f"Unknown learning rule: {learning_rule}")

        self.weights = self.weights + delta.astype(np.int32)
        self.weights = np.clip(self.weights, -self.L, self.L)

    def is_synchronized(self, other: 'TPM') -> bool:
        """Check whether two TPMs share identical weights."""
        if not isinstance(other, TPM):
            return False
        if (self.K, self.N, self.L) != (other.K, other.N, other.L):
            return False
        return np.array_equal(self.weights, other.weights)

    def export_key(self, key_length: Optional[int] = None) -> bytes:
        """Export TPM weights as a byte-based keystream."""
        shifted_weights = self.weights + self.L
        flat_weights = shifted_weights.flatten()

        if key_length is None:
            key_bytes = flat_weights.astype(np.uint8).tobytes()
        else:
            if len(flat_weights) >= key_length:
                key_bytes = flat_weights[:key_length].astype(np.uint8).tobytes()
            else:
                key_bytes = flat_weights.astype(np.uint8).tobytes()
                while len(key_bytes) < key_length:
                    expansion = (flat_weights ^ (flat_weights >> 1)).astype(np.uint8).tobytes()
                    key_bytes += expansion[:min(len(expansion), key_length - len(key_bytes))]

        return key_bytes[:key_length] if key_length else key_bytes


def synchronize(
    tpm_a: TPM,
    tpm_b: TPM,
    chaos_sequence: np.ndarray,
    max_steps: int = 10000,
    learning_rule: Literal["hebbian", "anti_hebbian", "random_walk"] = "hebbian",
    debug: bool = False
) -> Tuple[int, bytes, bytes]:
    """Synchronize two TPMs using a chaotic sequence for input generation."""
    K = tpm_a.K
    N = tpm_a.N
    input_size = K * N

    chaos_sequence = np.clip(chaos_sequence, 0.0, 1.0)

    if len(chaos_sequence) < input_size:
        repeats = (input_size // len(chaos_sequence)) + 1
        chaos_sequence = np.tile(chaos_sequence, repeats)

    steps = 0
    chaos_idx = 0
    no_update_streak = 0
    max_no_update = 2000

    agreement_count = 0
    total_agreements = 0
    binary_balance_stats = []

    while steps < max_steps:
        chaos_values = []
        for _ in range(input_size):
            if chaos_idx >= len(chaos_sequence):
                chaos_idx = 0
            chaos_values.append(chaos_sequence[chaos_idx])
            chaos_idx += 1

        chaos_values = np.array(chaos_values)

        sorted_indices = np.argsort(chaos_values)
        x_flat = np.ones(input_size, dtype=np.int32) * -1
        x_flat[sorted_indices[input_size // 2:]] = 1

        if input_size % 2 == 1:
            middle_idx = sorted_indices[input_size // 2]
            x_flat[middle_idx] = 1 if (steps + chaos_idx) % 2 == 0 else -1

        shuffle_seed = (steps + chaos_idx) % (2**31)
        np.random.seed(shuffle_seed)
        np.random.shuffle(x_flat)
        np.random.seed(None)

        x = x_flat.reshape(K, N)

        sigma_a, tau_a = tpm_a.predict(x)
        sigma_b, tau_b = tpm_b.predict(x)

        if tau_a == tau_b:
            tpm_a.update(x, tau_a, learning_rule)
            tpm_b.update(x, tau_b, learning_rule)
            no_update_streak = 0
            agreement_count += 1
            total_agreements += 1
        else:
            no_update_streak += 1

        if debug and steps % 1000 == 0:
            num_ones = np.sum(x_flat == 1)
            balance_ratio = num_ones / len(x_flat)
            binary_balance_stats.append(balance_ratio)

        steps += 1

        if no_update_streak >= max_no_update:
            alt_rule = "random_walk" if learning_rule == "hebbian" else "hebbian"
            for _ in range(min(50, max_steps - steps)):
                chaos_values = []
                for _ in range(input_size):
                    if chaos_idx >= len(chaos_sequence):
                        chaos_idx = 0
                    chaos_values.append(chaos_sequence[chaos_idx])
                    chaos_idx += 1

                chaos_values = np.array(chaos_values)
                sorted_indices = np.argsort(chaos_values)
                x_flat = np.ones(input_size, dtype=np.int32) * -1
                x_flat[sorted_indices[input_size // 2:]] = 1
                if input_size % 2 == 1:
                    middle_idx = sorted_indices[input_size // 2]
                    x_flat[middle_idx] = 1 if (steps + chaos_idx) % 2 == 0 else -1

                shuffle_seed = (steps + chaos_idx) % (2**31)
                np.random.seed(shuffle_seed)
                np.random.shuffle(x_flat)
                np.random.seed(None)

                x = x_flat.reshape(K, N)
                sigma_a, tau_a = tpm_a.predict(x)
                sigma_b, tau_b = tpm_b.predict(x)

                if tau_a == tau_b:
                    tpm_a.update(x, tau_a, alt_rule)
                    tpm_b.update(x, tau_b, alt_rule)
                    no_update_streak = 0
                    break

                steps += 1
                if steps >= max_steps:
                    break

        if tpm_a.is_synchronized(tpm_b):
            break

    if debug:
        try:
            import logging
            logger = logging.getLogger(__name__)
            sync_status = "OK" if tpm_a.is_synchronized(tpm_b) else "FAIL"
            agreement_pct = 100 * total_agreements / max(1, steps)
            logger.info(f"[TPM] Sync: {sync_status} | Steps: {steps}/{max_steps} | Agreement: {agreement_pct:.1f}%")
            if binary_balance_stats:
                avg_balance = np.mean(binary_balance_stats)
                logger.info(f"[TPM] Binary balance: {avg_balance:.4f} (ideal: 0.5)")
        except Exception:
            pass

    key_a = tpm_a.export_key()
    key_b = tpm_b.export_key()

    return steps, key_a, key_b
