import argparse
import gymnasium as gym
import torch

from stable_baselines3 import DQN
from car_racing import CarRacing


parser = argparse.ArgumentParser(description='PyTorch RL')
parser.add_argument('--network-arch', type=int, nargs = "*", default=[256, 256], metavar='NN',
                    help='neural network model size for training (default: [256, 256])')
args = parser.parse_args()
print("network architecture: {}".format(args.network_arch))

mode = "state"
env = CarRacing(render_mode = mode, continuous = False)
state = env.reset()

# SAC hyperparams:
model = DQN("MlpPolicy", env, verbose=1, policy_kwargs = {'net_arch': args.network_arch})
#  model = DQN("CnnPolicy", env, verbose=1)


#  num_step = 1e3
#  num_step = 1e4
#  num_step = 1e5
num_step = 1e6
#  num_step = 1e7
print(num_step)

model.learn(int(num_step))

traced_script_module = torch.nn.Sequential(model.q_net.features_extractor, model.q_net.q_net).cpu()
traced_script_module = torch.jit.script(traced_script_module)

file_name = "model_3_{}_5_{}.pt".format("_".join([str(i) for i in args.network_arch]), num_step)
traced_script_module.save(file_name)
