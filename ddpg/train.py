import argparse
import gym
import torch.nn.functional as F
from torch import optim
import torch
import rc_gym
import wandb
from ddpg import *
import time

if __name__ == "__main__":
    # A yaml with training paramters needs to be selected
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Select yaml file with experiment configs')
    args = parser.parse_args()

    wandb.init(
        project="paper_robocup", 
        entity="felipemartins", 
        config=args.config,
        save_code=True
    )
    
    hp = wandb.config
    wandb.run.name = hp.experiment_name + "_" + wandb.run.name

    # Create directories for saving networks and gifs
    os.makedirs(f"saves/{hp.experiment_name}/models/", exist_ok=True)
    os.makedirs(f"saves/{hp.experiment_name}/gifs/", exist_ok=True)

    # 
    device = torch.device(hp.torch_device)
    
    # Create env
    env = gym.make(hp.env)

    # Init networks
    obs_size, act_size = env.observation_space.shape[0], env.action_space.shape[0]
    pi = PolicyNet(obs_size, act_size).to(device)
    Q = QNet(obs_size, act_size).to(device)
    pi_tgt = TargetPi(pi)
    Q_tgt = TargetQ(Q)

    # Logging Networks
    wandb.watch(pi)
    wandb.watch(Q)

    # Network optimizers
    pi_opt = optim.Adam(pi.parameters(), lr=hp.learning_rate_pi)
    Q_opt = optim.Adam(Q.parameters(), lr=hp.learning_rate_q)

    # Replay Buffer
    buffer = Buffer(size=hp.buffer_size, device=device)
    
    # Action Noise
    noise = OUNoise(sigma=hp.noise_sigma_start, min_value=env.action_space.low,
                    max_value=env.action_space.high)

    # Training counters
    total_steps = 0
    total_eps = 0
    # Train indefinitely
    while True:
        S = env.reset()
        noise.reset()
        if noise.sigma > hp.noise_sigma_min:
            noise.sigma *= hp.noise_sigma_decay  # slowly decrease noise scale

        done = False
        ep_steps = 0
        ep_start_time = time.time()
        # Loop until end of episode
        while not done:
            metrics = {}
            with torch.no_grad():
                np_S = np.array(S, dtype=np.float32)
                A = noise(pi(torch.tensor(np_S).to(device)).cpu().numpy())
                S_tp1, r, done, info = env.step(A)
                
                ep_steps += 1
                if done:
                    metrics.update(info)
                    metrics['fps'] = ep_steps/(time.time() - ep_start_time)
                    metrics['noise_sigma'] = noise.sigma
                buffer.add(experience(S, A, r, done, S_tp1))
            S = S_tp1
            if len(buffer) < hp.replay_initial:
                wandb.log(metrics)
                continue

            S_v, A_v, r_v, dones_v, S_tp1_v = buffer.sample(hp.batch_size)

            # train Q
            Q_opt.zero_grad()
            Q_v = Q(S_v, A_v)
            A_tp1_v = pi_tgt.target_model(S_tp1_v)
            Q_tp1_v = Q_tgt.target_model(S_tp1_v, A_tp1_v)
            Q_tp1_v[dones_v] = 0.0
            Q_ref_v = r_v.unsqueeze(dim=-1) + Q_tp1_v*hp.gamma
            Q_loss_v = F.mse_loss(Q_v, Q_ref_v.detach())
            Q_loss_v.backward()
            Q_opt.step()
            metrics['loss_Q'] = Q_loss_v

            # Train pi
            pi_opt.zero_grad()
            A_cur_v = pi(S_v)
            pi_loss_v = -Q(S_v, A_cur_v)
            pi_loss_v = pi_loss_v.mean()
            pi_loss_v.backward()
            pi_opt.step()
            metrics['pi_loss'] = pi_loss_v

            pi_tgt.sync(hp.alpha)
            Q_tgt.sync(hp.alpha)

            # Save model
            if total_steps % hp.model_save_frequency == 0:
                torch.save(pi.state_dict(),
                           f"saves/{hp.experiment_name}/models/pi.pth")
                torch.save(Q.state_dict(),
                           f"saves/{hp.experiment_name}/models/Q.pth")
                wandb.save(f"saves/{hp.experiment_name}/models/pi.pth")
                wandb.save(f"saves/{hp.experiment_name}/models/Q.pth")
                
            # Save an episode gif
            if total_steps % hp.gif_frequency == 0:
                idx = total_steps / hp.gif_frequency
                generate_gif(
                    env, f"saves/{hp.experiment_name}/gifs/{idx:05d}.gif", pi, 1000, device)
                wandb.save(f"saves/{hp.experiment_name}/gifs/{idx:05d}.gif")

            wandb.log(metrics)
            total_steps += 1

        total_eps += 1

    wandb.join()
