import os
import pickle
import tensorflow as tf
import numpy as np
import gym
from gym import envs
import load_policy
import math
import run_expert

class Dagger:
    """
    Class to run dagger on a task
    """

    def __init__(self, task):
        self.supportedTasks = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]
        if task in self.supportedTasks:
            self.task = task
            self.expertPolicyFile = "./experts/" + self.task + ".pkl"
            self.expertDataFile = "./expert_data/" + self.task + ".pkl"
            self.policyDataFile = "./policy_data/" + self.task + ".pkl"
        else:
            print("Task %s does not exist" % (task))
            return
        print("Running dagger on " + str(self.task))

        self.env = gym.make(self.task)
        
        """
        if ! (isinstance(self.env.observation_space, gym.spaces.discrete.Discrete) or isinstance(self.env.observation_space, gym.spaces.box.Box)):
            print("Observation space type not currently supported")
            return

        if ! (isinstance(self.env.env.action_space, gym.spaces.discrete.Discrete) or isinstance(self.env.env.action_space, gym.spaces.box.Box)):
            print("Action space type not currently supported")
            return

        if isinstance(self.env.observation_space, gym.spaces.box.Box):
            self.obsSpaceShape = self.env.observation_space.shape
        elif isinstance(self.env.observation_space, gym.spaces.discrete.Discrete):
            # needed to one hot encode discrete space
            self.obsSpaceShape = (self.env.observation_space.n,)

        if isinstance(self.env.action_space, gym.spaces.box.Box)
            self.actionSpaceShape = self.env.action_space.shape
        elif isinstance(self.env.actionSpaceShape, gym.spaces.discrete.Discrete):
            # needed to one hot encode discrete space
            self.actionSpaceShape = (self.env.actionSpaceShape.n,)
        
        """

        self.obsSpaceShape = self.env.observation_space.shape
        self.actionSpaceShape = self.env.action_space.shape

        neuronsInHiddenLayer = [512, 256, 128, 64]
        activationsForHiddenLayer = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
       
        # sets self.outputTensor
        self.genModel(neuronsInHiddenLayer, activationsForHiddenLayer)
        self.session = tf.Session()
        self.policyAvgReturns = []
    
    def genModel(self, neuronsInHiddenLayer, activationsForHiddenLayer):
        """
        Sets output tensor, using vanilla neural network with dense layers

        """

        if len(neuronsInHiddenLayer) != len(activationsForHiddenLayer):
            print("neuronsInLayer and activationsForLayer dimensions do not match")
        
        obsTensorShape = (None, np.prod(self.env.observation_space.shape))
        obs = tf.placeholder(
                                dtype=tf.float32, 
                                shape=obsTensorShape, 
                                name="observation"
        )
        x = obs
        for i in range(len(neuronsInHiddenLayer)):
            name = "dense" + str(i) 
            x = tf.layers.dense(
                inputs=x, 
                units=neuronsInHiddenLayer[i], 
                activation=activationsForHiddenLayer[i],
                name=name
            )

        outputTensor = tf.layers.dense(
            inputs=x,
            units=np.prod(self.env.action_space.shape),
            name="action"
        )

        actionTensorShape = (None,) + self.env.action_space.shape
        expertAction = tf.placeholder(
            dtype=tf.float32, 
            shape=actionTensorShape,
            name="expertAction"
        )
        loss = tf.losses.mean_squared_error(outputTensor, expertAction)
        gradientStep = tf.train.AdamOptimizer().minimize(loss)

        self.outputTensor = outputTensor
        self.lossTensor = loss
        self.gradientStepOp = gradientStep

    def generateExpertData(self, numRollouts=20, render=False):
        print("Generating new expert data")
        run_expert.runExpertOnNewSamples(self.expertPolicyFile, self.task, render, self.env.spec.timestep_limit, numRollouts)      

    def getExpertData(self):
        with open(self.expertDataFile,'rb') as f:
            data = pickle.load(f)
        return (data["observations"], data["actions"])


    def getAction(self, observations):
        if self.session == None:
            print("session is None, cannot get action")
            return
        action = self.session.run(
            self.outputTensor, 
            feed_dict={
                "observation:0" : observations
        })
        return action

    def resetSession(self):
        self.session = tf.Session()

    def train(self, epochs=100, batchSize=100):
        if self.outputTensor == None or self.lossTensor == None or self.gradientStepOp == None:
            print("Model is not defined. Pleas define using genModel before training")
            return
        print("Training model")
        
        if self.session == None:
            self.session = tf.session()

        expertObservations, expertActions = self.getExpertData()
        
        expertObservations = np.array(expertObservations)
        expertActions = np.array(expertActions)

        expertObservations = np.reshape(expertObservations, (-1, np.prod(self.env.observation_space.shape)))
        expertActions = np.reshape(expertActions, (-1, np.prod(self.env.action_space.shape)))

        indexes = np.arange(len(expertObservations))
        shuffle = np.random.permutation(indexes)
        expertObservations = expertObservations[shuffle]
        expertActions = expertActions[shuffle]

        #split = int(0.8 * len(expertObservations))

        #expertObservationsTrain
 
        self.session.run(tf.global_variables_initializer())
        for i in range (epochs):
            losses = []
            for j in range(math.ceil(len(expertObservations) // batchSize)):
                begin = j * batchSize
                end = min(begin + batchSize, len(expertObservations) - 1)
                batch_loss, _ = self.session.run([self.lossTensor, self.gradientStepOp], feed_dict={
                    "observation:0" : expertObservations[begin:end],
                    "expertAction:0": expertActions[begin:end]
                })
                losses.append(batch_loss)
            print("Epoch " + str(i+1) + " loss: " + str(np.mean(losses)))

    def runPolicy(self, numRollouts, render=False):
        print("Running policy on environment")
        env = gym.make(self.task)
        max_steps = self.env.spec.timestep_limit

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
                action = self.getAction([obs])
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
        
        self.policyAvgReturns.append((np.mean(returns), np.std(returns)))

        policy_data = {'observations': np.array(observations), 
                        'actions': np.array(actions),
                        "returns" : np.array(self.policyAvgReturns)}

    
        with open(self.policyDataFile, 'wb') as f:
            pickle.dump(policy_data, f, pickle.HIGHEST_PROTOCOL)
      
    def runExpertOnPolicyData(self):
        print("Running expert on policy data")
        run_expert.runExpertOnObservations(self.expertPolicyFile, self.policyDataFile, self.task)

    def closeSession(self):
        self.session.close()

    def cloneExpertBehavior(self, initialExpertRollouts, epochs,  batchSize, policyEvaluationRollouts):
        self.generateExpertData(initialExpertRollouts)
        self.train(epochs, batchSize)
        self.runPolicy(policyEvaluationRollouts)
    
    
    def daggerLoop(self, initialExpertRollouts, loopIterations, policyRolloutsPerIteration, trainingEpochsPerLoop, batchSizePerLoop):
        self.cloneExpertBehavior(initialExpertRollouts, trainingEpochsPerLoop,  batchSizePerLoop, policyRolloutsPerIteration)
        for i in range(loopIterations):
            print("Dagger loop " + str(i+1))
            if i != 0:
                self.train(trainingEpochsPerLoop, batchSizePerLoop)
                self.runPolicy(policyRolloutsPerIteration)
            self.runExpertOnPolicyData()

if __name__ == "__main__":
    tasks = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]
    for task in tasks:
        print(task)
        dagger = Dagger(task)
        dagger.daggerLoop(10, 20, 10, 100, 128)
        dagger.closeSession()
   