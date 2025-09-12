# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=0 python gns/train.py --mode=train --data_path=./datasets/taylor_impact_2d/data_processed/ --model_path=./models/Taylor_impact_2d/ --output_path=./rollouts/Taylor_impact_2d/ --batch_size=4 --noise_std=0.02 --connection_radius=1.2 --layers=10 --hidden_dim=128 --lr_init=0.001 --ntraining_steps=100000 --lr_decay_steps=20000 --input_sequence_length=6 --project_name=Taylor_impact_2d --run_name=Seq6_R1.2_NS0.02_L10_N128 --nsave_steps=2000 -log=True

# Train msgnn
CUDA_VISIBLE_DEVICES=0 python gns/multi_scale/train_multi_scale.py --mode=train --data_path=./datasets/taylor_impact_2d/onecase_data_processed/ --model_path=./models/Taylor_impact_2d/ --output_path=./rollouts/Taylor_impact_2d/ --radius_multiplier=3 --batch_size=32 --noise_std=0.02 --layers=10 --hidden_dim=128 --lr_init=0.001 --ntraining_steps=10000 --lr_decay_steps=2000 --dim=2 --project_name=Taylor_impact_2d_OneCase --run_name=OneCase_RM3_NS0.02_L10_N128 --nsave_steps=2000 -log=True

# Rollout
CUDA_VISIBLE_DEVICES=0 python gns/multi_scale/train_multi_scale.py --mode=rollout --data_path=./datasets/taylor_impact_2d/onecase_data_processed/ --model_path=./models/Taylor_impact_2d/OneCase_RM3_NS0.02_L10_N128/ --model_file=model-010000.pt --output_path=./rollouts/Taylor_impact_2d/ --radius_multiplier=3 --noise_std=0.02 --layers=10 --hidden_dim=128 --dim=2 --project_name=Taylor_impact_2d --run_name=OneCase_RM3_NS0.02_L10_N128

# Visualisation
python -m gns.render_rollout_taylor_impact_2d --rollout_path=rollouts/Taylor_impact_2d/OneCase_RM3_NS0.02_L10_N128/ --output_path=rollouts/Taylor_impact_2d/OneCase_RM3_NS0.02_L10_N128/ --batch_mode=True


# Notes
- If net config changed before evaluation, load weights may fail
- If subtle config changed, evaluation may have low results
- Train loss (acc) and val loss (pos) are not comparable currently
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise)
- pytorch geometric caps the knn in radius_graph to be <=32
- The original domain is x (-165, 165) and y (-10, 85). Normalise it to (0,1) and (0,1) will change the w/h ratio. 
- Be careful with the simulation domain, the Bullet in impact loading has made y too large unnessarily
