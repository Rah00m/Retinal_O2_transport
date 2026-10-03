"""
Measure inference latency of the trained 4-network PINN, loading the
saved checkpoint instead of retraining.

Run:
    python timing_model.py
"""
import time
import numpy as np

import config
from networks import load_networks


def time_inference(fn, *args, num_runs=100):
    fn(*args)  # warm-up, excluded from timing
    times = np.empty(num_runs)
    for i in range(num_runs):
        t0 = time.perf_counter()  # perf_counter: higher resolution than time.time()
        fn(*args)
        times[i] = time.perf_counter() - t0
    return times.mean(), times.std()


def main():
    nets = load_networks(config.CHECKPOINT_DIR)
    z = np.linspace(0, config.LAYER_LENGTH_UM, config.N_GRID_POINTS_PER_LAYER)
    z = z.reshape(-1, 1).astype("float32")

    print(f"{'Layer':<6}{'Avg (ms)':<12}{'Std (ms)':<12}{'Per point (ms)':<16}")
    print("-" * 46)
    total_avg = 0.0
    for layer in config.LAYERS:
        avg, std = time_inference(nets[layer], z)
        total_avg += avg
        print(f"{layer:<6}{avg*1000:<12.4f}{std*1000:<12.4f}{avg/len(z)*1000:<16.4f}")

    print(f"\nSum of 4 separate calls (one per layer network): {total_avg*1000:.4f} ms")
    print("Note: unlike a single-network case, these 4 calls genuinely hit 4")
    print("distinct models -- there is no equivalent 'single combined call' here")
    print("since each network only accepts its own layer's local z domain.")


if __name__ == "__main__":
    main()