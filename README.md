# Pong
Using reinforcement learning to teach agent to win at Pong

Currently in progress to recreating this project.

<h2> Process:</h2>

<p1><br> a) Read and Understand the code. </p1></br>
<p2><br> b) Recreate and rewrite what I understood from scratch. </p2></br>
<p3><br> c) Train the neural network, may take around 3 days or more based on the value set for the hyperparameters </p3></br>

<p4></br> V1 - All comments belong to me and reflect my understand, and I updated the 3 image processing functions into one.<p4></br>
<br> Breakdown: </br>
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

<h3> References: </h3>
<p1>
<br>http://karpathy.github.io/2016/05/31/rl/</br>
<br>http://neuralnetworksanddeeplearning.com/</br>
<br>http://ruder.io/optimizing-gradient-descent/index.html#rmsprop</br>
<br>https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0</br>
</p1>
