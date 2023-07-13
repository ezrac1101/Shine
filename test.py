import argparse
import gymnasium as gym
import glob
import numpy as np
import time
import torch

network_name = []

#  network_name.extend(["16_32"])
network_name.extend(["3_256_256_5_1000.0"])
#  network_name.extend(["32_64"])
#  network_name.extend(["64_64"])
#  network_name.extend(["128_128"])
#  network_name.extend(["32_32_32"])
#  network_name.extend(["64_64_64"])


from car_racing import CarRacing
mode = "state"
env = CarRacing(render_mode = mode, continuous = False, shape = "random")
max_horizon = 3000

def test_dqn(model, episode, render = True):
    total_score = 0
    run_time = 0.0
    run_num = 0
    for e in range(episode):
        seed = np.random.randint(low = 0, high = np.iinfo(np.int32).max)
        state = env.reset(seed = seed)[0]
        #  print(state)
        #  input()
        score = 0
        idx = 0
        while True:
            t = time.time()
            state = torch.tensor(state).unsqueeze(0)

            if env.render_mode == "state_pixels":
                state = state.permute(0, 3, 1, 2).float()
            else: assert(env.render_mode in ("state", "human", None))
            #  print(state)
            #  input()

            if isinstance(model, torch.jit.ScriptModule):
                action = torch.argmax(model.forward(state)).item()
            elif isinstance(model, SoftDecisionTree):
                action = torch.argmax(model.predict(state)).item()
            else: assert(False)
            run_time += time.time() - t
            run_num += 1
            idx += 1
            #  print("state: {}".format(state))
            #  print(state.shape)
            #  print("action: {}".format(action))
            #  print(idx)
            #  input()
            #  action = 1 if state[2] >= 0 else 0
            #  print(env.step(action))
            state, reward, done1, done2, _ = env.step(action)
            #  print(done1)
            #  print(done2)
            #  input()
            if render:
                env.render()

            score += reward
            if done1 or done2 or idx >= max_horizon:
                #  print(idx)
                print("test: episode: {}/{}, score: {}".format(e, episode, score), end = '\r')
                #  print("test: episode: {}/{}, score: {}".format(e, episode, score))
                break
            #  print(state)
            #  print(action)
            #  print(reward)
            #  print(done)
            #  input()
        total_score += score
    print("    Average Score: {}".format(total_score / episode))
    print("    Average Inference Time: {}".format(run_time / run_num))
    return total_score / episode, run_time / run_num

#  model = ""
#  test_dqn(model, 1, True)


for n in network_name:
    print("model name: {}".format(n))
    file_name = []
    #  nn
    file_name.append("model_{}.pt".format(n))

    for f in file_name:
        print("  nn: {}".format(f))
        model = torch.jit.load(f)

        output = "{}".format(f[:-3])

        np.random.seed(0)
        _, run_time = test_dqn(model, 3, True)

env.close()
