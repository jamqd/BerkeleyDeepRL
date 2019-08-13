#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import math

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    return args

def runExpertOnObservations(expertPolicyFile, observationsFile, envName, batchSize=16):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expertPolicyFile)
    print('loaded and built')

    with open(os.path.join('policy_data', envName + '.pkl'), 'rb') as f:
        policy_data = pickle.load(f)
    
    observations = np.array(policy_data["observations"])

    with tf.Session():
        tf_util.initialize()

        expertActions = []
        print(len(observations))
        for i in range(len(observations)):
            action = policy_fn(observations[i][None,:])
            expertActions.append(action)
        
        try:
            with open(os.path.join('expert_data', envName + '.pkl'), 'rb') as f:
                expert_data = pickle.load(f)
            
            expert_data["observations"] = np.concatenate((expert_data["observations"], observations))
            expert_data["actions"] = np.concatenate((expert_data["actions"], expertActions))
            
        except:
            expertObservations = observations
            expert_data = {
                'observations': np.array(expertObservations),
                'actions': np.array(expertActions)
            }
        print("Now has " + str(len(expert_data["observations"])) + " expert observations and actions")
        with open(os.path.join('expert_data', envName + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
    return

def runExpertOnNewSamples(expertPolicyFile, envName, render, maxTimeSteps, numRollouts=20):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expertPolicyFile)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envName)
        max_steps = maxTimeSteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(numRollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       "returns" : np.array(returns),
                       "originalMeanReturn" : np.mean(returns),
                       "originalStdReturn" : np.std(returns)}
        print("Generated " + str(len(expert_data["observations"])) + " expert observations and actions")
        with open(os.path.join('expert_data', envName + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
    return

def main():
    args = parseArgs()
    runExpertOnNewSamples(args.expert_policy_file, args.envname, args.render, args.max_timesteps, args.num_rollouts)

if __name__ == '__main__':
    main()
