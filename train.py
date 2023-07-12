import argparse
import gymnasium as gym
import torch

from stable_baselines3 import DQN
from car_racing import CarRacing


parser = argparse.ArgumentParser(description='PyTorch RL')
parser.add_argument('--network-arch', type=int, nargs = "*", default=[32, 32], metavar='NN',
                    help='input batch size for training (default: [32, 32])')
args = parser.parse_args()
print("network architecture: {}".format(args.network_arch))

mode = "state"
env = CarRacing(render_mode = mode, continuous = False)
state = env.reset()

# SAC hyperparams:
model = DQN("MlpPolicy", env, verbose=1, policy_kwargs = {'net_arch': args.network_arch})
#  model = DQN("CnnPolicy", env, verbose=1)

#  m = model.get_parameters()
#  for k,i in m['policy'].items():
#      print(k, i.shape)
#  input()

num_step = 2e3
#  num_step = 2e7
model.learn(int(num_step))

traced_script_module = torch.nn.Sequential(model.q_net.features_extractor, model.q_net.q_net).cpu()
traced_script_module = torch.jit.script(traced_script_module)

file_name = "model_3_{}_5.pt".format("_".join([str(i) for i in args.network_arch]))
traced_script_module.save(file_name)
