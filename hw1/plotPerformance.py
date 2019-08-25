import matplotlib.pyplot as plt
import numpy as np 
import pickle


tasks = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]

imitationRewards = {}
imitationRewardsSTD = {}
imitationRewardsList = []
imitationRewardsSTDList = []


daggerRewards = {}
daggerRewardsSTD = {}
daggerRewardsList = []
daggerRewardsSTDList = []

expertRewards = {}
expertRewardsSTD = {}
expertRewardsList = []
expertRewardsSTDList = []

# get policy, dagger, and expert data
for task in tasks:
    with open("./imitation_policy_data/" + task + ".pkl", "rb") as f:
        data = pickle.load(f)
        imitationRewards[task] = np.array(data["returns"][:,0])
        imitationRewardsSTD[task] = np.array(data["returns"][:,1])
        imitationRewardsList.append(imitationRewards[task][-1])
        imitationRewardsSTDList.append(imitationRewardsSTD[task][-1])

    with open("./dagger_policy_data/" + task + ".pkl", "rb") as f:
        data = pickle.load(f)
        daggerRewards[task] = np.array(data["returns"][:,0])
        daggerRewardsSTD[task] = np.array(data["returns"][:,1])
        daggerRewardsList.append(daggerRewards[task][-1])
        daggerRewardsSTDList.append(daggerRewardsSTD[task][-1])

    with open("./expert_data/" + task + ".pkl", "rb") as f:
        data = pickle.load(f)
        expertRewards[task] = data["originalMeanReturn"]
        expertRewardsSTD[task] = data["originalStdReturn"]
        expertRewardsList.append(expertRewards[task])
        expertRewardsSTDList.append(expertRewardsSTD[task])

print(tasks)

print("Expert Rewards")
print(expertRewardsList)
print("Expert Rewards STDs")
print(expertRewardsSTDList)

print("Imitation Rewards")
print(imitationRewardsList)
print("Imitation Rewards STDs")
print(imitationRewardsSTDList)


print("Dagger Rewards")
print(daggerRewardsList)
print("Dagger Rewards STDs")
print(daggerRewardsSTDList)

# Plot reward vs task
plt.figure(figsize=(8,8))
x_pos = np.arange(len(tasks))
# policy plot
plt.errorbar(x_pos, imitationRewardsList, yerr=imitationRewardsSTDList, fmt='r.',mfc='red', label="Imitation Policy Performance")
# expert plot
plt.errorbar(x_pos, expertRewardsList, yerr=expertRewardsSTDList, fmt='b.',mfc='blue', label="Expert Policy Performance")

plt.xticks(x_pos, tasks)
plt.xlabel("Task")
plt.ylabel('Mean Reward')
plt.title('Rewards from Imitation Learning vs Expert by Task')
plt.legend(loc="best")
plt.savefig("./report/plots/imitationExpertComparison.png")


# Plot reward vs number of epochs
for i in range(len(tasks)):
    plt.figure()

    # policy performance plot
    epochs = np.arange(len(imitationRewards[tasks[i]]))
    plt.plot(epochs, imitationRewards[tasks[i]], "r", label="Imitation Policy Performance")

    # expert performance plot
    x_sample = np.linspace(1, len(epochs))
    plt.plot(x_sample, np.repeat(expertRewardsList[i], len(x_sample)), "b", label="Average Expert Policy Performance")

    plt.xlabel("Epoch")
    plt.ylabel("Mean Reward")
    plt.title(tasks[i] + " Reward vs Training Epochs (Imitation)")
    plt.legend(loc="best")
    plt.savefig("./report/plots/" + tasks[i] + "Imitation.png")




# plot reward vs number of dagger loops
for i in range(len(tasks)):
    plt.figure()

    # policy performance plot
    daggerLoops = np.arange(len(daggerRewards[tasks[i]]))
    plt.plot(daggerLoops, daggerRewards[tasks[i]], "g", label="Dagger Policy Performance")

    # expert performance plot
    x_sample = np.linspace(1, len(daggerLoops))
    plt.plot(x_sample, np.repeat(daggerRewardsList[i], len(x_sample)), "b", label="Average Expert Policy Performance")

    # imitation performace plot
    plt.plot(x_sample, np.repeat(imitationRewardsList[i], len(x_sample)), "r", label="Average Imitation Policy Performance")

    plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
    plt.xlabel("Dagger Loop")
    plt.ylabel("Mean Reward")
    plt.title(tasks[i] + " Reward vs Iterations (Dagger)")
    plt.legend(loc="best")
    plt.savefig("./report/plots/" + tasks[i] + "Dagger.png")


# plot imitation, dagger, and expert comparison across tasks

plt.figure(figsize=(8,8))
x_pos = np.arange(len(tasks))
# policy plot
plt.errorbar(x_pos, imitationRewardsList, yerr=imitationRewardsSTDList, fmt='r.',mfc='red', label="Imitation Policy Performance")
# expert plot
plt.errorbar(x_pos, expertRewardsList, yerr=expertRewardsSTDList, fmt='b.',mfc='blue', label="Expert Policy Performance")
# dagger plot
plt.errorbar(x_pos, daggerRewardsList, yerr=daggerRewardsSTDList, fmt='g.',mfc='green', label="Dagger Policy Performance")

plt.xticks(x_pos, tasks)
plt.xlabel("Task")
plt.ylabel('Mean Reward')
plt.title('Rewards from Imitation Learning vs Dagger vs Expert by Task')
plt.legend(loc="best")
plt.savefig("./report/plots/overallComparison.png")

#plt.show()



