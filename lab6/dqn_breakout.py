"""DLP DQN Lab"""
__author__ = "chengscott"
__copyright__ = "Copyright 2020, NCTU CGI Lab"
import argparse
from collections import deque
import itertools
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity):
        self.index = 0
        self.size = 0
        self.capacity = capacity
        self.states = torch.zeros((capacity, 5, 84, 84), dtype=torch.uint8)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(self, state, action, reward, done):
        """Saves a transition"""
        self.states[self.index] = state
        self.actions[self.index, 0] = action
        self.rewards[self.index, 0] = reward
        self.dones[self.index, 0] = done
        self.index = (self.index + 1) % self.capacity
        self.size = max(self.size, self.index)

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        if self.capacity < batch_size:
            raise ValueError("Can not sample")
        num = torch.randint(0, high=self.size, size=(batch_size,))

        states = self.states[num, :4].to(device)
        actions = self.actions[num].to(device)
        rewards = self.rewards[num].to(device)
        dones = self.dones[num].to(device)
        next_states = self.states[num, 1:].to(device)
        return states, actions, rewards.float(), next_states, dones.float()

    def __len__(self):
        return self.size


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(True), nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.0
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(device)
        self._target_net = Net().to(device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(
            self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4
        )

        ## TODO ##
        """Initialize replay buffer"""
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        """epsilon-greedy based on behavior network"""
        ## TODO ##
        if random.random() >= epsilon:
            with torch.no_grad():
                actions = self._behavior_net(state)
                action = torch.argmax(actions).reshape(1, 1)
                return action.cpu()
        else:
            action = action_space.sample()
        return action

    def append(self, state, action, reward, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        self._memory.push(state, action, reward / 10, int(done))

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size)
        ## TODO ##
        q_value = torch.gather(self._behavior_net(state), dim=1, index=action.long())
        with torch.no_grad():
            q_next = torch.max(self._target_net(next_state), dim=1)[0].unsqueeze(dim=1)
            q_target = reward + gamma * (1 - done) * q_next
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        self.clip_gradient(self._optimizer, 1)
        nn.utils.clip_grad_value_(self._behavior_net.parameters(), 1)
        self._optimizer.step()

    def _update_target_network(self):
        """update target network by copying from behavior network"""
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    "behavior_net": self._behavior_net.state_dict(),
                    "target_net": self._target_net.state_dict(),
                    "optimizer": self._optimizer.state_dict(),
                },
                model_path,
            )
        else:
            torch.save(
                {
                    "behavior_net": self._behavior_net.state_dict(),
                },
                model_path,
            )

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model["behavior_net"])
        if checkpoint:
            self._target_net.load_state_dict(model["target_net"])
            self._optimizer.load_state_dict(model["optimizer"])


def frame(frames):
    frames = torch.Tensor(frames)
    frames = frames.reshape((1, frames.size(-2), frames.size(-2)))
    return frames


def train(args, agent, writer):
    print("Start Training")
    env_raw = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env_raw, episode_life=True, clip_rewards=True)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.0
    ewma_reward = 0
    queue = deque(maxlen=5)

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        state, reward, done, _ = env.step(1)  # fire first !!!
        for _ in range(10):
            result = env.step(0)
            frames = frame(result[0])
            queue.append(frames)

        for t in itertools.count(start=1):
            state = torch.cat(list(queue))[1:].unsqueeze(0).to(device)
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                # select action
                action = agent.select_action(state, epsilon, action_space)
                # decay epsilon
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)
            # execute action
            frames, reward, done, _ = env.step(action)

            frames = frame(frames)
            queue.append(frames)

            ## TODO ##
            # store transition
            agent.append(torch.cat(list(queue)).unsqueeze(0), action, reward, done)

            total_reward += reward
            if total_steps >= args.warmup:
                agent.update(total_steps)

            if total_steps % args.eval_freq == 0 and total_steps != 0:
                """You can write another evaluate function, or just call the test function."""
                test(args, agent, writer)
                agent.save(args.model + "dqn_" + str(total_steps) + ".pt")

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar("Train/Episode Reward", total_reward, episode)
                writer.add_scalar("Train/Ewma Reward", ewma_reward, episode)
                print(
                    "Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}".format(
                        total_steps, episode, t, total_reward, ewma_reward, epsilon
                    )
                )
                break
    env.close()


def test(args, agent, writer):
    print("Start Testing")
    env_raw = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env_raw)
    action_space = env.action_space
    e_rewards = []
    queue = deque(maxlen=5)
    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        done = False
        for _ in range(5):  # no-op
            result = env.step(0)
            frames = frame(result[0])
            queue.append(frames)
        done = result[2]

        while not done:
            state = torch.cat(list(queue))[len(queue) - 4 :].unsqueeze(0).to(device)

            time.sleep(0.01)
            action = agent.select_action(state, args.test_epsilon, action_space)
            result = env.step(action)
            frames = frame(result[0])
            queue.append(frames)
            reward = result[1]
            e_reward += reward
            done = result[2]
        writer.add_scalar("DQN_breakout_Test/Episode Reward", e_reward, i)
        print("episode {}: {:.2f}".format(i + 1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print(
        "Average Reward: {:.2f}".format(
            float(sum(e_rewards)) / float(args.test_episode)
        )
    )


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", default="ckpt/")
    parser.add_argument("--logdir", default="log/dqn_breakout")
    # train
    parser.add_argument("--warmup", default=20000, type=int)
    parser.add_argument("--episode", default=55000, type=int)
    parser.add_argument("--capacity", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.0000625, type=float)
    parser.add_argument("--eps_decay", default=1000000, type=float)
    parser.add_argument("--eps_min", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--freq", default=4, type=int)
    parser.add_argument("--target_freq", default=10000, type=int)
    parser.add_argument("--eval_freq", default=200000, type=int)
    # test
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("-tmp", "--test_model_path", default="ckpt/dqn_1000000.pt")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--test_episode", default=10, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--test_epsilon", default=0.01, type=float)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
    else:
        train(args, agent, writer)


if __name__ == "__main__":
    main()
