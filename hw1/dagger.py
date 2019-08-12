import os
import pickle
import tensorflow as tf
import numpy as np
import gym
from gym import envs
import load_policy

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

        neuronsInHiddenLayer = [32,16]
        activationsForHiddenLayer = [tf.nn.relu, tf.nn.relu]
       
        # sets self.outputTensor
        self.genModel(neuronsInHiddenLayer, activationsForHiddenLayer)

        self.session = tf.Session()
    
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
            activation=tf.nn.relu,
            name="action"
        )
        self.outputTensor = outputTensor

    def getExpertData(self):
        with open(os.path.join('expert_data', self.task + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        return (data["observations"], data["actions"])


    def getAction(self, observations):
        action = self.session.run(
            self.outputTensor, 
            feed_dict={
                "observation:0" : observations
        })
        return action

    def resetSession(self):
        self.session = tf.Session()

    def train(self, epochs=100, batchSize=100):
        if self.outputTensor == None:
            print("Model is not defined. Pleas define using genModel before training")
            return
        obsTensorShape = (None,) + self.env.action_space.shape
        expertAction = tf.placeholder(
            dtype=tf.float32, 
            shape=obsTensorShape,
            name="expertAction"
        )
        loss = tf.losses.mean_squared_error(self.outputTensor, expertAction)
        gradientStep = tf.train.AdamOptimizer().minimize(loss)
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
            for j in range(len(expertObservations) // batchSize):
                begin = i * batchSize
                end = min(begin + batchSize, len(expertObservations) - 1)
                batch_loss, _ = self.session.run([loss, gradientStep], feed_dict={
                    "observation:0" : expertObservations[begin:end],
                    "expertAction:0": expertActions[begin:end]
                })
                losses.append(batch_loss)
            print("Epoch " + str(i+1) + " loss: " + str(np.mean(losses)))

    def evaluatePolicy(self):

        return

    def closeSession(self):
        self.session.close()

if __name__ == "__main__":
    test = Dagger("Ant-v2")
    test.train(3, 64)
    test.closeSession()
    #expertObservations, expertActions = test.getExpertData()
    #print(expertObservations.shape)
    #print(expertActions.shape)
    