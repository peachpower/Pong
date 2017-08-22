## Problem: 
	# Given a sequence of images, determine the move that will result in winning the game. 

## Solution 
	# Take the images from the game, process them to remove color, background and down sample. it.
	# Send these images to the neural net, to compute the probability of moving up.
	# Take the probability and find the probability of moving down.
	# Send this action to the agent.
	# Record if this action results in a win or lose.
	# After 21 rounds are over, pass the result to compute gradient.
	# Take an average of 10 gradient values and move the weights so they reflect the new value.
	# Repeat this until you beat the computer. 
#Import the openai gym
import gym 
import numpy as np

## Updated this function 
def preprocess_image(image):
	# Down sample,take every other pixel. Note: it reduces the resolution of the image.
	image = image[::2, ::2, :]
	# Remove color, where variable is passed [R,G,B], remove R and G.
	image = image[:,:,0]
	# Remove background, since we only need to know where our ball and agent is.
	image[image == 144] = 0
	image[image == 109] = 0
	return image


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    processed_observation = input_observation[35:195] # crop
	processed_observation = preprocess_image(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # Subtract the previous frame from the current one so we are only processing changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # Store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations

# Sigmoid function takes the value and returns a valid probability. 
# Such that it is between 0 and 1. 
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# Rectified Linear Unit, analogous with half-wave rectification, matrix-vector product
# Speeds up training by easing gradient computation, sets it to 0 or 1
def relu(vector):
    vector[vector < 0] = 0
    return vector
	
# Applying Feature Learning techniques, dot product and ReLu operations to both input and output values
# Note: sigmoid is being applied to the output_layer_values where it is between 0 and 1 and this value is
# being passed as a probability of moving up
def apply_neural_nets(observation_matrix, weights):
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values
	
# Sending the command to move up or down based on calculated probability of moving up
def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # moving up, based on openai documentation
        return 2
    else:
         # moving down, based on openai documentation
        return 3

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) 

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


def main():
    env = gym.make("Pong-v0")
    observation = env.reset() #Acquiring the image

    ## A set of Hyperparameters
	# Counter to track the number of rounds played
    round_number = 0
	# Weights are adjusted after the round_number = batch_size
    batch_size = 10
	# the impact of previous actions on future actions
    gamma = 0.99 # discount factor for reward
	# Changes the function (in  our case rmsprop algorithm)
	# to ensure that we avoid over fitting
    decay_rate = 0.99
	# The number of neurons in our neural network
    num_hidden_layer_neurons = 200
	# The dimensions of our image
    input_dimensions = 80 * 80
	# Learning rate is how quickly the neural net adjusts to
	# new training scenarios. Intuitively a higher learning 
	# rate reduces the amount of time spent training the net.
	# In practise this can cause the net to generalize and jump
	# to conclusions (and generalizations) that are not correct. 
    learning_rate = 1e-4

    round_number = 0
    reward_sum = 0
    running_reward = None
	# Initially starting with no information about the image
    prev_processed_observations = None

	# weights['1'] - Matrix that holds weights of pixels passing into hidden layer. Dimensions: [200 x 80 x 80] -> [200 x 6400]
	# weights['2'] - Matrix that holds weights of hidden layer passing into output. Dimensions: [1 x 200] 
		# Dot product between weights[1] (200x6400 matrix)and the observation_matrix(6400x1 matrix) = output (200x1 matrix). 
    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    # reference made in article: To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

	# Initializing arrays to store observations about the game 
    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

	# While the simulation is running, save the image at time 0 and time 1
	# Downsample the image, remove color, and remove the background
	# Send these values to the neural net and calculate the new output layer values
	# Set the previous state image to be the current image
	# Send the move action to the agent
    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
    
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # reference made:  http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)


        if done: 
            round_number += 1

            
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

           
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
              episode_gradient_log_ps_discounted,
              episode_hidden_layer_values,
              episode_observations,
              weights
            )

            
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if round_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] 
            observation = env.reset() 
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            reward_sum = 0
            prev_processed_observations = None

main()