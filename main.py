import gym
from gym import wrappers
from learners import *

EPSILON = 1 * 10 ** -6
learner = None

cartpole = {    "name": 'CartPole-v0',
                "buckets": (10, 10, 8, 8), # Discretization buckets
                "limits": (4.8, 10, 0.42, 5)
            }

mountaincar = {    "name": 'MountainCar-v0',
                "buckets": (15, 15, 10, 10), # Discretization buckets
                "limits": (4.8, 10, 0.42, 5)
            }

env, BUCKETS, LIMITS = None, None, None
def make_env(params):
    global env
    global BUCKETS
    global LIMITS
    env = gym.make(params['name'])
    env = wrappers.Monitor(env, "/tmp/" + params['name'], force = True)
    BUCKETS, LIMITS = params['buckets'], params['limits']

def main():
    global learner
    make_env(cartpole)
    # there's a bug with the unmonitored envs not checking max_steps
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    TIMESTEP_MAX = max_steps

    # no_drop is a reward function for cartpole that penalises falling before 200. accelerates learning massively.
    no_drop = lambda fail, success: lambda r, t: fail if t != TIMESTEP_MAX else success
    identity = lambda r, x: r

    # Epsilon Greedy Double Deep Q-Learner
    learner = DoubleDeepQLearner(len(env.observation_space.high), 
                                (8, 16, 32, env.action_space.n),
                                256, 
                                100000, 
                                identity,
                                lambda learner: EpsilonGreedy(learner, 1, 10 ** -5))

    # hash function
    def discretizer(observation):
        b, l = BUCKETS, LIMITS
        # EPSILON used to keep within bucket bounds
        bounded = [min(l[i] - EPSILON,max(-l[i] + EPSILON, observation[i])) for i in range(len(observation))]
        return tuple(math.floor((bounded[i] + l[i]) / 2 / l[i] * b[i]) for i in range(len(bounded)))

    # CountBasedOptimism Double Deep Q-Learner
    learner = DoubleDeepQLearner(len(env.observation_space.high), (8, 16, env.action_space.n),
                                 256, 
                                 100000, 
                                 lambda r, x: r,
                                 lambda learner: CountBasedOptimism(learner, 0.3, -10 ** -5, discretizer, 10))

    # VDBE Double Deep Q-Learner
    learner = DoubleDeepQLearner(len(env.observation_space.high), 
                                 (8, 16, env.action_space.n),
                                 256, 
                                 100000, 
                                 lambda r, x: r,
                                 lambda learner: VDBE(learner, 0.3, discretizer, 2, 1/env.action_space.n))

    total = 0
    for i_episode in range(10000):
        s1 = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = learner.act(s1)
            s0 = s1
            s1, reward, done, info = env.step(action)
            learner.learn(s0, s1, reward, action, done, t+1)
            ep_reward += reward
            if done: break
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(i_episode, t+1, total/(i_episode+1)))
    env.close()

    #gym.upload('/tmp/cartpole0', api_key='')
main()


