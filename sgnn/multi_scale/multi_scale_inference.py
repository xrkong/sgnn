import os
import pickle
import sys
from pathlib import Path

import torch

# Ensure project root is on path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.resource_monitor import ResourceMonitor
from sgnn.multi_scale.static_graph_data_loader import (
    get_multi_scale_data_loader_by_trajectories,
)
from sgnn.multi_scale import multi_scale_evaluate as validate_multi_scale


def run_inference(simulator, metadata, config, device):
    """Run multi-scale rollout inference and (optionally) save per-case outputs.

    Expects config keys: model_path, run_name, model_file, output_path,
    data_path, input_sequence_length, inference_mode, dim, num_scales,
    window_size, radius_multiplier, mode ('rollout' or 'valid').
    """
    monitor = ResourceMonitor(device)

    # Load model weights
    try:
        model_path = Path(config['model_path']) / config['run_name'] / config['model_file']
        simulator.load(str(model_path))
        print(f"✅ Loaded model from: {model_path}")
    except Exception as e:
        print(
            f"❌ Failed to load model weights from {config['model_path']}{config['run_name']}/{config['model_file']}"
        )
        print(f"Error: {e}")
        sys.exit(1)

    simulator.to(device)
    simulator.eval()

    # Ensure output directory exists
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    # Choose split
    split = 'test' if config['mode'] == 'rollout' else 'valid'

    data_trajs = get_multi_scale_data_loader_by_trajectories(
        path=str(Path(config['data_path']) / f'{split}.npz'),
        num_scales=config['num_scales'],
        window_size=config['window_size'],
        radius_multiplier=config['radius_multiplier'],
    )

    eval_loss = []
    max_memory_overall = 0.0
    total_time = 0.0

    with torch.no_grad():
        for example_i, data_traj in enumerate(data_trajs):
            monitor.start()

            nsteps = metadata['sequence_length'] - config['input_sequence_length']
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)

            # Set static graph for this trajectory
            simulator.set_static_graph(data_traj['graph'])

            # Predict rollout
            example_output = validate_multi_scale.evaluate_multi_scale_rollout(
                simulator=simulator,
                positions=positions,
                particle_type=particle_type,
                n_particles_per_example=n_particles_per_example,
                strains=strains,
                nsteps=nsteps,
                dim=config['dim'],
                device=device,
                input_sequence_length=config['input_sequence_length'],
                inference_mode=config['inference_mode'],
            )

            example_output['metadata'] = metadata

            # Loss summaries
            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
            loss_position = example_output['rmse_position'][-1]
            loss_strain = example_output['rmse_strain'][-1]

            stats = monitor.stop()
            total_time += stats['elapsed_time']
            max_memory_overall = max(max_memory_overall, stats['max_memory_mb'])

            print(
                f"Predicting example {example_i}-\n"
                f"      {example_output['metadata']['file_valid'][example_i]}\n"
                f"      loss_total: {loss_total}, loss_position: {loss_position}, loss_strain: {loss_strain}"
            )
            print(f"  Runtime: {stats['elapsed_time']:.2f}s, VRAM: {stats['max_memory_mb']:.1f}MB")
            eval_loss.append(loss_total)

            # Save rollout in testing
            if config['mode'] == 'rollout':
                simulation_name = metadata['file_test'][example_i]
                case_name = simulation_name.replace('.npz', '')
                example_output['case_name'] = case_name

                save_dir = Path(config['output_path']) / config['run_name']
                save_dir.mkdir(parents=True, exist_ok=True)
                filename = save_dir / f'{case_name}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(example_output, f)

    # Summary
    print("\n" + "=" * 70)
    print("📊 Rollout Benchmark Summary")
    print("=" * 70)
    print(f"Mean loss: {sum(eval_loss) / len(eval_loss):.6f}")
    print(f"Total runtime: {total_time:.2f}s")
    print(f"Average runtime per rollout: {total_time / max(1, len(eval_loss)):.2f}s")
    print(f"Peak VRAM usage: {max_memory_overall:.1f}MB")
    print("=" * 70)


