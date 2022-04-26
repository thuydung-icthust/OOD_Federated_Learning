from ddpg_agent.utils import *
from ddpg_agent.networks import *
from ddpg_agent.policy import *
from ddpg_agent.buffer import *
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from ddpg_agent.policy import NormalizedActions
import wandb
from utils import *
from defense import *
class DDPG_Agent(nn.Module):
    def __init__(
        self,
        state_dim=3,
        action_dim=1,
        device="cuda",
        hidden_dim=256,
        init_w=1e-3,
        value_lr=1e-3,
        policy_lr=1e-4,
        replay_buffer_size=1000000,
        max_steps=16*50,
        max_frames=12000,
        batch_size=4,
        beta=0.45,
        log_dir="./log/epochs",
        gamma = 0.99,
        soft_tau = 2e-2,
    ):
        super(DDPG_Agent, self).__init__()
        self.lr = 3e-2
        self.num_steps = 20  # number of iterations for each episodes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.actions = []
        self.states = []

        # self.policy_network = policy_network

        self.log_probs = []
        self.values = []
        self.rewards = []  # rewards for each episode
        # self.masks     = []
        self.entropy = 0
        self.step_count = 0
        self.frame_idx = 0
        self.max_frames = max_frames  # number of episodes
        self.episode_reward = 0
        self.beta = beta # coefficient for mean and std losses inside reward func
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        new_action_space = spaces.Box(low=0, high=1, shape=(action_dim,))
        self.ou_noise = OUNoise(new_action_space)

        print("Init State dim", state_dim)
        print("Init Action dim", action_dim)

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(self.device).double() # 30 + 30 = 60 as input
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device).double()

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(self.device).double()
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device).double()

        # store all the (s, a, s', r) during the transition process
        self.memory = Memory()
        # replay buffer used for main training
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # self.model = ActorCritic(num_inputs, num_outputs, hidden_size)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.value_lr  = 1e-3
        # self.policy_lr = 1e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()
        self.step = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_dir = log_dir

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)


    def ddpg_update(self, min_value=-np.inf, max_value=np.inf):

        for i in range(int(len(self.replay_buffer)/self.batch_size)):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            # state = torch.DoubleTensor(state).squeeze().cuda()
            # next_state = torch.DoubleTensor(next_state).squeeze().cuda()
            # action = torch.DoubleTensor(action).squeeze().cuda()
            # reward = torch.DoubleTensor(reward).cuda()
            # done = torch.DoubleTensor(np.float32(done)).cuda()

            state = torch.DoubleTensor(state).squeeze().to(self.device)
            next_state = torch.DoubleTensor(
                next_state).squeeze().to(self.device)
            action = torch.DoubleTensor(action).squeeze().to(self.device)
            reward = torch.DoubleTensor(reward).to(self.device)
            done = torch.DoubleTensor(np.float32(done)).to(self.device)

            policy_loss = self.value_net(state, self.policy_net(state), self.batch_size)
            policy_loss = -policy_loss.mean()
            next_action = self.target_policy_net(next_state)
            target_value = self.target_value_net(next_state, next_action.detach(), self.batch_size)

            expected_value = reward + (1.0 - done) * self.gamma * target_value.squeeze()
            expected_value = torch.clamp(expected_value, min_value, max_value)

            value = self.value_net(state, action, self.batch_size).squeeze()

            value_loss = self.value_criterion(value, expected_value)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()


        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)


    def get_action(self, client_models, pseudo_avg, num_dps, net_avg, net_freq, g_user_indices, round, prev_reward):
        # reach to maximum step for each episode or get the done for this iteration
        # state = get_state(start_loss = start_loss, final_loss = final_loss, std_local_losses=std_local_losses,epochs=local_num_epochs, num_samples=local_n_samples, clients_id=clients_id)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        # base_model = net_avg
        # load_model_weight(base_model, torch.from_numpy(net_avg.astype(np.float32)).to(device)) 
        total_client = len(client_models)
        done=0.0
        # print("OK")
        # bias_list, weight_list, avg_bias, avg_weight, weight_update = extract_classifier_layer(client_models, net_avg, prev_net=
        bias_list, weight_list, weight_update, prev_avg_weight, prev_avg_bias = extract_classifier_layer_new(client_models, net_avg)
        # print("OK2")
        
        w_eucl_dis, w_cs_dis = get_distance_on_avg_net(weight_list, prev_avg_weight, weight_update, total_client)
        b_eucl_dis = get_ed_on_base_net(bias_list, prev_avg_bias, total_client)
        
        state = extract_state(w_eucl_dist=w_eucl_dis, w_cos_dist=w_cs_dis, b_eucl_dist=b_eucl_dis)
        print(state)
        print("state.shape is: ", state.shape)
        if self.frame_idx >= self.max_frames:
            # maybe stop training?
            self.logging_per_round()
            state = self.reset_state()

        # state = torch.DoubleTensor(state).unsqueeze(0).cuda()  # current state
        state = torch.DoubleTensor(state).unsqueeze(0).to(self.device)  # current state
        print("OK2")
        
        if prev_reward:
            self.memory.update(r=prev_reward)

        action = self.policy_net.get_action(state)
        action = self.ou_noise.get_action(action, self.step)
        self.memory.act(state, action)
        print("action: ", action)

        # if self.step < self.max_steps:
        if self.memory.get_last_record() is None:
            self.step += 1
            self.frame_idx += 1
            pred_attackers = get_pred_attackers(action)

            selected_client_idxs = [i for i in range(total_client) if i not in pred_attackers]
            print(f"selected_client: {selected_client_idxs}")
        
            selected_num_dps = np.array(num_dps)[selected_client_idxs]
            reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]
            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            vectorize_nets = np.asarray(vectorize_nets)[selected_client_idxs]
        
            aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

            aggregated_model = client_models[0] # slicing which doesn't really matter
        
        
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(self.device)) 
            neo_net_list = [aggregated_model]
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, aggregated_grad, vectorize_nets         
            # 
            # return action
        

        s, a, r, s_next = self.memory.get_last_record()
        self.replay_buffer.push(s, a, r, s_next, done)
        # print("OK4")

        if len(self.replay_buffer) >= self.batch_size:
            self.ddpg_update()

        self.episode_reward += prev_reward
        self.frame_idx += 1
        self.step += 1
        # print(action)

        pred_attackers = get_pred_attackers(action)
        print("OK5")
        
        selected_client_idxs = [i for i in range(total_client) if i not in pred_attackers]
        print(f"selected_client: {selected_client_idxs}")
        
        selected_num_dps = np.array(num_dps)[selected_client_idxs]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]
        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        vectorize_nets = np.asarray(vectorize_nets)[selected_client_idxs]
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        
        
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(self.device)) 
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, aggregated_grad, vectorize_nets         
        # return action

    def reset_state(self):
        self.ou_noise.reset()
        self.episode_reward = 0
        self.step = 0
        # self.memory.reset()
        # self.replay_buffer.reset()
        return np.zeros(self.state_dim)

    def logging_per_round(self):
        frame_idx, episode, total_reward = (
            self.frame_idx,
            self.episode_reward,
            self.episode_reward,
        )
        sample = {
            "frame_idx": frame_idx,
            "episode": episode,
            "total_reward": total_reward,
        }
        filename = self.log_dir + "/log_dqn.txt"
        with open(filename, "a+") as log_f:
            log_f.write(str(sample))


if __name__ == "__main__":
    max_frames = 12000
    max_steps = 16
    frame_idx = 0
    rewards = []
    batch_size = 128
    # Test simulator
    env = NormalizedActions(gym.make("Pendulum-v0"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 10
    agent = DDPG_Agent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
    )
    ou_noise = OUNoise(env.action_space)
    reward, done = None, None

    while frame_idx < max_frames:
        state = env.reset()
        ou_noise.reset()
        # episode_reward = 0

        for step in range(max_steps):
            # action = policy_net.get_action(state)
            # action = ou_noise.get_action(action, step)
            action = agent.get_action(state, reward, done)
            next_state, reward, done, _ = env.step(action)

            # replay_buffer.push(state, action, reward, next_state, done)
            # if len(replay_buffer) > batch_size:
            #     ddpg_update(batch_size)

            state = next_state

            # episode_reward += reward
            # frame_idx += 1

            # if frame_idx % max(1000, max_steps + 1) == 0:
            #     plot(frame_idx, rewards)

            # if done:
            #     break

        # rewards.append(episode_reward)
