import json
import os
import sys
import torch
import pickle
import yaml
import argparse
import time
from pathlib import Path

import wandb

# Add project root to path (go up 3 levels from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sgnn.single_scale import learned_simulator
from sgnn import noise_utils
from utils import reading_utils
from datasets.taylor_impact_2d.taylor_impact_data_loader import (
    get_data_loader_by_samples,
    get_data_loader_by_trajectories
)
from sgnn.single_scale import evaluate
from utils.resource_monitor import ResourceMonitor
from utils.checkpoint_utils import load_model as ckpt_load_model

# Load configuration from file
def load_config(config_path):
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    # Resolve relative paths
    if not config_file.is_absolute():
        # Try relative to current working directory first
        if config_file.exists():
            config_file = config_file.resolve()
        # Otherwise try relative to this script
        elif (Path(__file__).parent / config_file).exists():
            config_file = (Path(__file__).parent / config_file).resolve()
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Global config - will be loaded in main()
config = None

KINEMATIC_PARTICLE_ID = -1  # Not used for Taylor Impact; kept for backward-compat


def predict(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device: str):
    """Predict rollouts.
  
    Args:
      simulator: Trained simulator if not will exit.
      metadata: Metadata for test set.
  
    """
    # Initialize resource monitor
    monitor = ResourceMonitor(device)
    
    # Load simulator
    try:
        model_path = Path(config['model_path']) / config['run_name'] / config['model_file']
        simulator.load(str(model_path))
        print(f"✅ Loaded model from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model weights from {config['model_path']}{config['run_name']}/{config['model_file']}")
        print(f"Error: {e}")
        sys.exit(1)

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if config['mode'] == 'rollout' else 'valid'

    data_trajs = get_data_loader_by_trajectories(
        path=str(Path(config['data_path']) / f'{split}.npz'))

    eval_loss = []
    max_memory_overall = 0
    total_time = 0
    
    with torch.no_grad():
        for example_i, data_traj in enumerate(data_trajs):
            # Start monitoring this rollout
            monitor.start()
            
            nsteps = metadata['sequence_length'] - config['input_sequence_length']
            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
            positions = data_traj['positions'].to(device)
            particle_type = data_traj['particle_type'].to(device)
            strains = data_traj['strains'].to(device)
                            
            # Predict example rollout
            example_output = evaluate.rollout(simulator,
                                              positions,
                                              particle_type,
                                              n_particles_per_example,
                                              strains,
                                              nsteps,
                                              config['dim'],
                                              device,
                                              config['input_sequence_length'],
                                              config['inference_mode'])

            example_output['metadata'] = metadata

            # RMSE loss with shape (time,)
            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
            loss_position = example_output['rmse_position'][-1]
            loss_strain = example_output['rmse_strain'][-1]
            loss_oneStep = example_output['rmse_position'][0] + example_output['rmse_strain'][0]  

            # Stop monitoring and get stats
            stats = monitor.stop()
            total_time += stats['elapsed_time']
            max_memory_overall = max(max_memory_overall, stats['max_memory_mb'])
            
            print(f'''Predicting example {example_i}-
                  {example_output['metadata']['file_valid'][example_i]} 
                  loss_total: {loss_total}, 
                  loss_position: {loss_position}, 
                  loss_strain: {loss_strain}''')
            print(f"  Runtime: {stats['elapsed_time']:.2f}s, VRAM: {stats['max_memory_mb']:.1f}MB")
            eval_loss.append(loss_total)

            # Save rollout in testing
            if config['mode'] == 'rollout':
                example_output['metadata'] = metadata
                # Use the actual case name from metadata instead of generic rollout_i
                simulation_name = metadata['file_test'][example_i]
                # Remove .npz extension and create .pkl filename
                case_name = simulation_name.replace('.npz', '')
                
                # Store the actual case name used for this rollout
                example_output['case_name'] = case_name
                
                filename = f'{case_name}.pkl'
                
                # Create subfolder using run_name, similar to model saving
                save_dir = Path(config['output_path']) / config['run_name']
                save_dir.mkdir(parents=True, exist_ok=True)
                
                filename = save_dir / filename
                with open(filename, 'wb') as f:
                    pickle.dump(example_output, f)

    print("\n" + "="*70)
    print("📊 Rollout Benchmark Summary")
    print("="*70)
    print(f"Mean loss: {sum(eval_loss) / len(eval_loss):.6f}")
    print(f"Total runtime: {total_time:.2f}s")
    print(f"Average runtime per rollout: {total_time / len(eval_loss):.2f}s")
    print(f"Peak VRAM usage: {max_memory_overall:.1f}MB")
    print("="*70)
                  
                  
def load_model(simulator, device):
    """Wrapper using utils.checkpoint_utils to load model and optimizer state.

    Returns (simulator, step, optimizer)
    """
    model_dir = config['model_path'] + config['run_name'] + '/'
    simulator, step, optimizer = ckpt_load_model(
        simulator,
        model_dir,
        config['model_file'],
        config['train_state_file'],
        device,
    )
    return simulator, step, optimizer
    

def train(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device: str):
    """Train the model.
  
    Args:
      simulator: Get LearnedSimulator.
    """
    # Initialize resource monitor
    monitor = ResourceMonitor(device)
    max_train_memory = 0
    max_val_memory = 0
    
    optimizer = torch.optim.Adam(simulator.parameters(), lr=config['lr_init'])
    step = 0
    # If model_path does not exist create new directory and begin training.
    model_path = config['model_path']
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # If model_path does exist and model_file and train_state_file exist continue training.
    if config['model_file'] is not None:
        simulator, step, optimizer = load_model(simulator, device)
        
    simulator.train()
    simulator.to(device)

    data_samples = get_data_loader_by_samples(path=str(Path(config['data_path']) / 'train.npz'),
                                            input_length_sequence=config['input_sequence_length'],
                                            batch_size=config['batch_size'],
                                            )
    
    print(f"🚀 Starting single-scale GNN training...")
    print(f"   - Layers: {config['layers']}, Hidden dim: {config['hidden_dim']}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Training steps: {config['ntraining_steps']}")
    print(f"   - Learning rate: {config['lr_init']}")
    print(f"   - Loss weights: Position={config['loss_weight_position']}, Strain={config['loss_weight_strain']}")
    
    step = 0
    not_reached_nsteps = True
    lowest_eval_loss = float('inf')
    
    try:
        while not_reached_nsteps:
            for data_sample in data_samples:
                log = {}  # wandb logging
                # Move data to device
                position = data_sample['input']['positions'].to(device)
                particle_type = data_sample['input']['particle_type'].to(device)
                n_particles_per_example = data_sample['input']['n_particles_per_example'].to(device)
                next_position = data_sample['output']['next_position'].to(device)
                next_strain = data_sample['output']['next_strain'].to(device)
                
                # Sample noise for training
                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                    position, noise_std_last_step=config['noise_std']
                ).to(device)

                # Forward pass
                optimizer.zero_grad()
                
                pred_acc, target_acc, pred_strain = simulator.predict_accelerations(
                    next_positions=next_position,
                    position_sequence_noise=sampled_noise,
                    position_sequence=position,
                    nparticles_per_example=n_particles_per_example,
                    particle_types=particle_type
                )

                # Calculate the loss
                loss_pos = (pred_acc - target_acc) ** 2
                loss_xy = loss_pos.mean(axis=0)  # for log purpose

                # if 1d, compute loss on x-axis only
                loss_pos = loss_pos.sum(dim=-1)

                # Calculate strain loss
                loss_strain = (pred_strain - next_strain) ** 2
                
                # Apply loss weights and average
                loss = config['loss_weight_position'] * loss_pos + config['loss_weight_strain'] * loss_strain
                loss = loss.mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                lr_new = config['lr_init'] * (config['lr_decay'] ** (step / config['lr_decay_steps'])) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new
                
                step += 1
                
                # WandB logging
                log["train/loss"] = loss
                log["train/loss-position"] = loss_pos.mean()
                log["train/loss-strain"] = loss_strain.mean()
                log["train/loss-x"] = loss_xy[0]
                log["train/loss-y"] = loss_xy[1]
                if config['dim'] == 3:
                    log["train/loss-z"] = loss_xy[2]
                log["lr"] = lr_new
                
                # Track memory usage
                current_memory = monitor.get_current_memory()
                max_train_memory = max(max_train_memory, current_memory)
                
                if step % 10 == 0:
                    print(f"Step {step}: Total Loss = {loss.item():.6f}, Position Loss = {loss_pos.mean().item():.6f}, Strain Loss = {loss_strain.mean().item():.6f}, VRAM: {current_memory:.1f}MB")

                # Validate periodically and save only if better
                if step % config['nsave_steps'] == 0 and step > 0:
                    save_dir = Path(config['model_path']) / config['run_name']
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Full validation during training
                    print(f"🔍 Running full validation at step {step}...")
                    simulator.eval()
                    
                    # Monitor validation
                    val_start_time = time.time()
                    
                    # Load validation trajectories
                    data_trajs = get_data_loader_by_trajectories(
                        path=str(Path(config['data_path']) / 'valid.npz')
                    )
                    
                    eval_loss_total, eval_loss_position, eval_loss_strain, eval_loss_oneStep = [], [], [], []
                    with torch.no_grad():
                        for example_i, data_traj in enumerate(data_trajs):
                            # Track validation memory
                            val_memory = monitor.get_current_memory()
                            max_val_memory = max(max_val_memory, val_memory)
                            nsteps = metadata['sequence_length'] - config['input_sequence_length']
                            n_particles_per_example = data_traj['n_particles_per_example'].to(device)
                            positions = data_traj['positions'].to(device)
                            particle_type = data_traj['particle_type'].to(device)
                            strains = data_traj['strains'].to(device)
                            
                            # Predict example rollout
                            example_output = evaluate.rollout(simulator,
                                                              positions,
                                                              particle_type,
                                                              n_particles_per_example,
                                                              strains,
                                                              nsteps,
                                                              config['dim'],
                                                              device,
                                                              config['input_sequence_length'],
                                                              config['inference_mode'])
                            
                            example_output['metadata'] = metadata

                            # RMSE loss with shape (time,)
                            loss_total = example_output['rmse_position'][-1] + example_output['rmse_strain'][-1]
                            loss_position = example_output['rmse_position'][-1]
                            loss_strain = example_output['rmse_strain'][-1]
                            loss_oneStep = example_output['rmse_position'][0] + example_output['rmse_strain'][0]  
                            
                            print(f'''Predicting example {example_i}-
                                  {example_output['metadata']['file_valid'][example_i]} 
                                  loss_total: {loss_total}, 
                                  loss_position: {loss_position}, 
                                  loss_strain: {loss_strain}''')
                            print(f"Prediction example {example_i} takes {example_output['run_time']}")
                            eval_loss_total.append(loss_total)
                            eval_loss_position.append(loss_position)
                            eval_loss_strain.append(loss_strain)
                            eval_loss_oneStep.append(loss_oneStep)
                        
                        eval_loss_mean = sum(eval_loss_total) / len(eval_loss_total)
                        val_time = time.time() - val_start_time
                        print(f"Mean loss on valid-set rollout prediction: {eval_loss_mean}. Current lowest eval loss is {lowest_eval_loss}.")
                        print(f"  Validation runtime: {val_time:.2f}s, VRAM: {max_val_memory:.1f}MB")
                        
                        # Save only if better than previous best
                        if eval_loss_mean < lowest_eval_loss:
                            lowest_eval_loss = eval_loss_mean
                            print(f"✅ Better model obtained! Saving checkpoint (val_loss: {eval_loss_mean:.6f})")
                            
                            # Save model
                            simulator.save(str(save_dir / f'model-best-{step:06}.pt'))
                            
                            # Save training state
                            train_state = dict(
                                optimizer_state=optimizer.state_dict(), 
                                global_train_state={"step": step, "lowest_eval_loss": lowest_eval_loss}
                            )
                            torch.save(train_state, str(save_dir / f'train_state-best-{step:06}.pt'))
                            print(f"💾 Model and training state saved at step {step}")
                        else:
                            print(f"⚠️  No improvement (current: {eval_loss_mean:.6f}, best: {lowest_eval_loss:.6f})")
                        
                        # Log validation metrics
                        log["val/loss"] = sum(eval_loss_total) / len(eval_loss_total)
                        log["val/loss-position"] = sum(eval_loss_position) / len(eval_loss_position)
                        log["val/loss-strain"] = sum(eval_loss_strain) / len(eval_loss_strain)
                        log["val/rmse-oneStep"] = sum(eval_loss_oneStep) / len(eval_loss_oneStep)
                        log["val/runtime"] = val_time
                        log["val/vram_mb"] = max_val_memory
                    
                    # Set back to training mode
                    simulator.train()
                
                if config['log']:
                    log["train/vram_mb"] = current_memory
                    wandb.log(log, step=step)
                    
                if step >= config['ntraining_steps']:
                    not_reached_nsteps = False
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Final summary
    save_dir = Path(config['model_path']) / config['run_name']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Only save fallback model if no validation was performed (no best model saved)
    if lowest_eval_loss == float('inf'):
        print("\n⚠️  No validation performed during training - saving final model as fallback")
        simulator.save(str(save_dir / f'model-final-{step:06}.pt'))
        train_state = dict(
            optimizer_state=optimizer.state_dict(), 
            global_train_state={"step": step}
        )
        torch.save(train_state, str(save_dir / f'train_state-final-{step:06}.pt'))
        print(f"💾 Fallback model saved to {save_dir}")
    else:
        print(f"\n✅ Training completed! Best model saved at validation loss: {lowest_eval_loss:.6f}")
        print(f"📁 Model location: {save_dir}")
    
    # Print benchmark summary
    print("\n" + "="*70)
    print("📊 Training Benchmark Summary")
    print("="*70)
    print(f"Peak training VRAM: {max_train_memory:.1f}MB")
    print(f"Peak validation VRAM: {max_val_memory:.1f}MB")
    print("="*70)


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
    """Instantiates the simulator.
  
    Args:
      metadata: JSON object with metadata.
      acc_noise_std: Acceleration noise std deviation.
      vel_noise_std: Velocity noise std deviation.
      device: PyTorch device 'cpu' or 'cuda'.
    """

    # Normalization stats
    normalization_stats = {
        'acceleration': {
            'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['acc_std']) ** 2 +
                              acc_noise_std ** 2).to(device),
        },
        'velocity': {
            'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['vel_std']) ** 2 +
                              vel_noise_std ** 2).to(device),
        },
    }
    
    # Compute number of particle types dynamically from metadata
    num_particle_types = metadata.get('num_particle_types', 1)
    print(f"Detected {num_particle_types} particle types from metadata")
    
    # Calculate node input features
    # (input_sequence_length-1) velocity timesteps * dim + wall distance + particle type embedding (if multiple types)
    nnode_in = (config['input_sequence_length'] - 1) * config['dim'] + 1
    if num_particle_types > 1:
        nnode_in += config['particle_type_embedding_size']
    
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=config['dim'],  # xyz
        nnode_in=nnode_in,  # (input_sequence_length-1) velocity timesteps * dim + wall distance
        nedge_in=config['dim'] + 1,    # input edge features, relative displacement in all dims + distance between two nodes
        latent_dim=config['hidden_dim'],
        nmessage_passing_steps=config['layers'],
        nmlp_layers=1,
        mlp_hidden_dim=config['hidden_dim'],
        connectivity_radius=config['connection_radius'],
        normalization_stats=normalization_stats,
        nparticle_types=num_particle_types,
        particle_type_embedding_size=config['particle_type_embedding_size'],
        device=device)
    
    print(f"✅ LearnedSimulator created:")
    print(f"   - Particle dimensions: {config['dim']}")
    print(f"   - Node input features: {nnode_in}")
    print(f"   - Edge input features: {config['dim'] + 1}")
    print(f"   - Latent dimension: {config['hidden_dim']}")
    print(f"   - Message passing steps: {config['layers']}")
    print(f"   - Connectivity radius: {config['connection_radius']}")

    return simulator


def main():
    """Train or evaluates the model."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Single-Scale GNN Training')
    parser.add_argument('--config', type=str, default='sgnn/single_scale/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'valid', 'rollout'],
                       help='Override mode from config file (train/valid/rollout)')
    parser.add_argument('--model_file', type=str,
                       help='Override model_file from config (required for valid/rollout modes)')
    parser.add_argument('--log', type=str, choices=['True', 'False'],
                       help='Override log setting from config (True/False)')
    args = parser.parse_args()
    
    # Load configuration
    global config
    try:
        config = load_config(args.config)
        print(f"✅ Loaded config from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Override config with command-line arguments
    if args.mode is not None:
        config['mode'] = args.mode
        print(f"   Mode overridden to: {args.mode}")
    
    if args.model_file is not None:
        config['model_file'] = args.model_file
        print(f"   Model file overridden to: {args.model_file}")
    
    if args.log is not None:
        config['log'] = args.log == 'True'
        print(f"   Log setting overridden to: {config['log']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    # Load metadata
    metadata_path = Path(config['data_path']) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create simulator
    simulator = _get_simulator(metadata, config['noise_std'], config['noise_std'], device)
    simulator.to(device)
    
    if config['mode'] == 'train':
        # Init wandb
        if config['log']:
            wandb.init(project=config['project_name'], name=config['run_name'])
            train(simulator, metadata, device)
            wandb.finish()
        else:
            train(simulator, metadata, device)
    elif config['mode'] in ['valid', 'rollout']:
        predict(simulator, metadata, device)


if __name__ == '__main__':
    main()
