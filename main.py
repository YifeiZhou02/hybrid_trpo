import wandb
import argparse
from itertools import count

import gym
import d4rl
import scipy.optimize
import os

import torch
from models import *
from replay_memory import Memory, ReplayBuffer
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import typing

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--exp_name', default="Vanilla TRPO")
parser.add_argument('--no_wandb', default=False, action='store_true')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)
off_dataset = d4rl.qlearning_dataset(env)
# print(off_dataset.keys())
not_dones = 1 - off_dataset['terminals']
off_buffer = ReplayBuffer(off_dataset['observations'], off_dataset['actions'],
                          off_dataset['rewards'], off_dataset['next_observations'], not_dones)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
q_net = Value(num_inputs+num_actions).cuda()


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


def update_params(batch, off_buffer):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))
    next_states = torch.Tensor(np.array(batch.next_state))
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0), 1)
    deltas = torch.Tensor(actions.size(0), 1)
    advantages = torch.Tensor(actions.size(0), 1)

    off_states = torch.Tensor(off_dataset['observations'])
    off_next_states = torch.Tensor(off_dataset['next_observations'])
    off_rewards = torch.Tensor(off_dataset['rewards'])
    off_actions = torch.Tensor(off_dataset['actions'])

    off_states, off_actions, off_rewards, off_next_states, off_not_dones = off_buffer.sample(
        off_buffer.capacity)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * \
            prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * \
            args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    # bellman targets
    with torch.no_grad():
        targets_ = (value_net((next_states)) + rewards).detach().cuda()
        targets.requires_grad = False
        off_targets = (value_net((off_next_states)) +
                       off_rewards).detach().cuda()
        off_targets.requires_grad = False
    targets = targets.cuda()
    off_states_actions = Variable(
        torch.cat([off_states, off_actions], dim=1)).cuda()

    def get_q_loss():
        # set_flat_params_to(q_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        # print(Variable(torch.cat([states, actions], dim=1)).size())
        qs_ = q_net(Variable(torch.cat([states, actions], dim=1)).cuda())
        # print(qs_.size())
        q_loss = (qs_ - targets).pow(2).mean()

        # weight decay
        for param in q_net.parameters():
            q_loss += param.pow(2).sum() * args.l2_reg

        # bellman error from online samples
        b_loss = (qs_ - targets_).pow(2).mean()
        q_loss += b_loss

        # bellman error for offline data
        off_b_loss = ((q_net(off_states_actions) -
                      off_targets).pow(2)*off_not_dones.cuda()).mean()
        q_loss += off_b_loss
        q_loss.backward()
        return q_loss
        # return (q_loss.data.double().numpy(), get_flat_grad_from(q_net).data.double().numpy())

    # flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
    #     get_q_loss, get_flat_params_from(q_net).double().numpy(), maxiter=25)
    # set_flat_params_to(q_net, torch.Tensor(flat_params))
    optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    for _ in range(25):
        optimizer.zero_grad()
        get_q_loss()
        optimizer.step()

    advantages = (advantages - advantages.mean()) / advantages.std()
    with torch.no_grad():
        off_qs = q_net(off_states_actions).cpu().data.clone()
        off_vs = value_net(off_states).data.clone()
        off_advantages = off_qs - off_vs
    off_advantages = (off_advantages - off_advantages.mean()
                      ) / off_advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(
        Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    off_action_means, off_action_log_stds, off_action_stds = policy_net(
        Variable(off_states))
    off_fixed_log_prob = normal_log_density(Variable(
        off_actions), off_action_means, off_action_log_stds, off_action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(
                    Variable(states))
                off_action_means, off_action_log_stds, off_action_stds = policy_net(
                    Variable(off_states))
        else:
            action_means, action_log_stds, action_stds = policy_net(
                Variable(states))
            off_action_means, off_action_log_stds, off_action_stds = policy_net(
                Variable(off_states))

        log_prob = normal_log_density(
            Variable(actions), action_means, action_log_stds, action_stds)
        off_log_prob = normal_log_density(Variable(
            off_actions), off_action_means, off_action_log_stds, off_action_stds)
        # add offline states

        action_loss = (-Variable(advantages) *
                       torch.exp(log_prob - Variable(fixed_log_prob))).mean()
        action_loss += (-Variable(off_advantages) *
                        torch.exp(off_log_prob - Variable(off_fixed_log_prob))).mean()
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + \
            (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
if args.no_wandb:
    os.environ['WANDB_MODE'] = 'offline'
wandb.init(
    project=args.env_name,
    config=vars(args),
    name=args.exp_name)
env_steps = 0
for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array(
                [action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()
    update_params(batch, off_buffer)

    env_steps += num_steps

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
    wandb.log({"env_steps": env_steps,
               "episode_reward": reward_batch})
