import numpy as np
import matplotlib.pyplot as plt
import math


def explore_then_commit_2_arm(arm1_mean, arm2_mean, T, aEpsilon): 
    """_summary_

    Args:
        arm1_mean (float): The mean value of the beurnoulli distribution for arm 1
        arm2_mean (float): The mean value of the beurnoulli distribution for arm 2
        T (_type_): The total number of trials to be made  including both exploration and exploitation
        aEpsilon (float): The epsilon value to be used in the epsilon-greedy algorithm

    Returns:
        averageReward (float): The average reward received from the two arms in T trials repeated 1000 times
    """
    
    armMeans = np.zeros(2)
    armMeans[0] = arm1_mean
    armMeans[1] = arm2_mean
    
    armTotals = np.zeros(2)
    armSampleAverages = np.zeros(2)
    
    arm1Count = 0
    arm2Count = 0
    
    totalReward = 0
    
    averageReward = 0
    
    for i in range(T):
        
        if np.random.binomial(1, aEpsilon) == 1:
            # randomly choose an arm
            arm = np.random.randint(2)

        else:
            # choose the greedy arm
            arm = np.argmax(armSampleAverages)
            
        if arm == 0:
            arm1Count += 1
        else:
            arm2Count += 1
            
        reward = np.random.binomial(1, armMeans[arm])
        totalReward += reward
        armTotals[arm] += reward
        if arm1Count != 0: 
            armSampleAverages[0] = armTotals[0] / arm1Count
        if arm2Count != 0:
            armSampleAverages[1] = armTotals[1] / arm2Count
        
    averageReward = totalReward / T
    
    return averageReward

def main():
    """
    We want to find the value of epsilon that maximizes the average reward for the 2-armed bandit problem. 
    To do this, we will run the explore_then_commit_2_arm function 1000 times for each value of epsilon in 
    the range [0, 1] and calculate the average reward for each epsilon value.
    """
    
    T = 500
    
    N_array = np.array([1, 6, 11, 16, 21, 26, 31, 36, 41, 46])
    
    epsilon_array = N_array * 2 / T
    
    num = 1000
    
    averageReward_array = np.zeros(len(epsilon_array))
    
    for j in range(num):
        for i in range(len(epsilon_array)):
            averageReward_array[i] += explore_then_commit_2_arm(0.4, 0.8, T, epsilon_array[i])
    
    averageReward_array /= num
    
    print(f"the maximum average reward is {np.max(averageReward_array)} when epsilon = {epsilon_array[np.argmax(averageReward_array)]} which corresponds to N = {N_array[np.argmax(averageReward_array)]}")
    
    plt.plot(epsilon_array, averageReward_array, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("average reward")
    plt.title("Average Reward vs. Epsilon")
    plt.xticks( epsilon_array, rotation=90 )
    plt.show()
    
        
    
    
    
    print("Execution Finished.")


if __name__ == "__main__":
    main()