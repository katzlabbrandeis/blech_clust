# Import stuff
from pomegranate import *
import numpy as np
import multiprocessing as mp
import math

def poisson_hmm_implement(n_states, threshold, max_iterations, seeds, n_cpu, binned_spikes, off_trials, edge_inertia, dist_inertia, hmm_type):

	# Create a pool of asynchronous n_cpu processes running poisson_hmm() - no. of processes equal to seeds
	pool = mp.Pool(processes = n_cpu)
	if hmm_type == 'generic':
		results = [pool.apply_async(poisson_hmm, args = (n_states, threshold, max_iterations, binned_spikes, seed, off_trials, edge_inertia, dist_inertia,)) for seed in range(seeds)]
	else:
		results = [pool.apply_async(poisson_hmm_feedforward, args = (n_states, threshold, max_iterations, binned_spikes, seed, off_trials, edge_inertia, dist_inertia,)) for seed in range(seeds)]

	output = [p.get() for p in results]

	# Remove any threads that didn't converge - these will have NaN's as the log likelihood
	cleaned_output = []
	for i in range(len(output)):
		if math.isnan(output[i][1]):
			continue
		else:
			cleaned_output.append(output[i])

	# Find the process that ended up with the highest log likelihood, and return it as the solution. If several processes ended up with the highest log likelihood, just pick the earliest one
	log_probs = [cleaned_output[i][1] for i in range(len(cleaned_output))]
	maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
	return cleaned_output[maximum_pos]

def multinomial_hmm_implement(n_states, threshold, max_iterations, seeds, n_cpu, binned_spikes, off_trials, edge_inertia, dist_inertia):

	# Create a pool of asynchronous n_cpu processes running multinomial_hmm() - no. of processes equal to seeds
	pool = mp.Pool(processes = n_cpu)
	results = [pool.apply_async(multinomial_hmm, args = (n_states, threshold, max_iterations, binned_spikes, seed, off_trials, edge_inertia, dist_inertia,)) for seed in range(seeds)]
	output = [p.get() for p in results]

	# Remove any threads that didn't converge - these will have NaN's as the log likelihood
	cleaned_output = []
	for i in range(len(output)):
		if math.isnan(output[i][1]):
			continue
		else:
			cleaned_output.append(output[i])

	# Find the process that ended up with the highest log likelihood, and return it as the solution. If several processes ended up with the highest log likelihood, just pick the earliest one
	log_probs = [cleaned_output[i][1] for i in range(len(cleaned_output))]
	maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
	return cleaned_output[maximum_pos]

def poisson_hmm(n_states, threshold, max_iterations, binned_spikes, seed, off_trials, edge_inertia, dist_inertia):

	# Seed the random number generator
	np.random.seed(seed)

	# Make a pomegranate HiddenMarkovModel object
	model = HiddenMarkovModel('%i' % seed)
	states = []
	# Make a pomegranate independent components distribution object and represent every unit with a Poisson distribution - 1 for each state
	for i in range(n_states):
		#emission_slice = (int((float(i)/n_states)*binned_spikes.shape[1]), int((float(i+1)/n_states)*binned_spikes.shape[1]))
		#initial_emissions = np.mean(binned_spikes[off_trials, emission_slice[0]:emission_slice[1], :], axis = (0, 1))
		states.append(State(IndependentComponentsDistribution([PoissonDistribution(np.random.rand()) for unit in range(binned_spikes.shape[2])]), name = 'State%i' % (i+1)))

	model.add_states(states)
	# Add transitions from model.start to each state (equal probabilties)
	for state in states:
		model.add_transition(model.start, state, float(1.0/len(states)))

	# Add transitions between the states - 0.97 is the probability of not transitioning in every state
	for i in range(n_states):
		not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
		for j in range(n_states):
			if i==j:
				model.add_transition(states[i], states[j], not_transitioning_prob)
			else:
				model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))

	# Bake the model
	model.bake()

	# Train the model only on the trials indicated by off_trials
	model.fit(binned_spikes[off_trials, :, :], algorithm = 'baum-welch', stop_threshold = threshold, max_iterations = max_iterations, edge_inertia = edge_inertia, distribution_inertia = dist_inertia, verbose = False)
	log_prob = [model.log_probability(binned_spikes[i, :, :]) for i in off_trials]
	log_prob = np.sum(log_prob)

	# Set up things to return the parameters of the model - the state emission and transition matrix
	state_emissions = []
	state_transitions = np.exp(model.dense_transition_matrix())
	for i in range(n_states):
		state_emissions.append([model.states[i].distribution.parameters[0][j].parameters[0] for j in range(binned_spikes.shape[2])])
	state_emissions = np.array(state_emissions)

	# Get the posterior probability sequence to return
	posterior_proba = []
	for i in range(binned_spikes.shape[0]):
		c, d = model.forward_backward(binned_spikes[i, :, :])
		posterior_proba.append(d)
	posterior_proba = np.exp(np.array(posterior_proba))

	# Get the json representation of the model - will be needed if we need to reload the model anytime
	model_json = model.to_json()

	return model_json, log_prob, 2*((n_states)**2 + n_states*binned_spikes.shape[2]) - 2*log_prob, (np.log(len(off_trials)*binned_spikes.shape[1]))*((n_states)**2 + n_states*binned_spikes.shape[2]) - 2*log_prob, state_emissions, state_transitions, posterior_proba

def poisson_hmm_feedforward(n_states, threshold, max_iterations, binned_spikes, seed, off_trials, edge_inertia, dist_inertia):

	# Seed the random number generator
	np.random.seed(seed)

	# Make a pomegranate HiddenMarkovModel object
	model = HiddenMarkovModel('%i' % seed)
	states = []
	# Make a pomegranate independent components distribution object and represent every unit with a Poisson distribution - 1 for each state
	for i in range(n_states):
		emission_slice = (int((float(i)/n_states)*binned_spikes.shape[1]), int((float(i+1)/n_states)*binned_spikes.shape[1]))
		initial_emissions = np.mean(binned_spikes[off_trials, emission_slice[0]:emission_slice[1], :], axis = (0, 1))
		states.append(State(IndependentComponentsDistribution([PoissonDistribution(initial_emissions[unit]) for unit in range(binned_spikes.shape[2])]), name = 'State%i' % (i+1)))

	model.add_states(states)
	# Add a transition from model.start to state 0 with p = 1, all other transitions from model.start are p = 0
	for i in range(len(states)):
		if i == 0:
			model.add_transition(model.start, states[i], 1.0)
		else:
			model.add_transition(model.start, states[i], 0.0)

	# Add transitions between the states - 0.95-0.99 is the probability of not transitioning in every state. Also transitions can only happen from state 0 to 1 to 2 to 3 and so on
	for i in range(n_states):
		not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
		for j in range(n_states):
			if i==j:
				model.add_transition(states[i], states[j], not_transitioning_prob)
			elif j - i == 1:
				model.add_transition(states[i], states[j], 1.0 - not_transitioning_prob)
			elif i == n_states - 1:
				model.add_transition(states[i], model.end, 1.0 - not_transitioning_prob)
			else:
				model.add_transition(states[i], states[j], 0.0)

	# Bake the model
	model.bake()

	# Train the model only on the trials indicated by off_trials
	model.fit(binned_spikes[off_trials, :, :], algorithm = 'baum-welch', stop_threshold = threshold, max_iterations = max_iterations, edge_inertia = edge_inertia, distribution_inertia = dist_inertia, verbose = False)
	log_prob = [model.log_probability(binned_spikes[i, :, :]) for i in off_trials]
	log_prob = np.sum(log_prob)

	# Set up things to return the parameters of the model - the state emission and transition matrix
	state_emissions = []
	state_transitions = np.exp(model.dense_transition_matrix())
	for i in range(n_states):
		state_emissions.append([model.states[i].distribution.parameters[0][j].parameters[0] for j in range(binned_spikes.shape[2])])
	state_emissions = np.array(state_emissions)

	# Get the posterior probability sequence to return
	posterior_proba = []
	for i in range(binned_spikes.shape[0]):
		c, d = model.forward_backward(binned_spikes[i, :, :])
		posterior_proba.append(d)
	posterior_proba = np.exp(np.array(posterior_proba))

	# Get the json representation of the model - will be needed if we need to reload the model anytime
	model_json = model.to_json()

	return model_json, log_prob, 2*((n_states)**2 + n_states*binned_spikes.shape[2]) - 2*log_prob, (np.log(len(off_trials)*binned_spikes.shape[1]))*((n_states)**2 + n_states*binned_spikes.shape[2]) - 2*log_prob, state_emissions, state_transitions, posterior_proba

def multinomial_hmm(n_states, threshold, max_iterations, binned_spikes, seed, off_trials, edge_inertia, dist_inertia):

	# Seed the random number generator
	np.random.seed(seed)

	# Make a pomegranate HiddenMarkovModel object
	model = HiddenMarkovModel('%i' % seed)
	states = []
	# Make a pomegranate Discrete distribution object with emissions = range(n_units + 1) - 1 for each state
	n_units = int(np.max(binned_spikes))
	for i in range(n_states):
		dist_dict = {}
		prob_list = np.random.random(n_units + 1)
		prob_list = prob_list/np.sum(prob_list)
		for unit in range(n_units + 1):
			dist_dict[unit] = prob_list[unit]
		states.append(State(DiscreteDistribution(dist_dict), name = 'State%i' % (i+1)))

	model.add_states(states)
	# Add transitions from model.start to each state (equal probabilties)
	for state in states:
		model.add_transition(model.start, state, float(1.0/len(states)))

	# Add transitions between the states - 0.95-0.999 is the probability of not transitioning in every state
	for i in range(n_states):
		not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
		for j in range(n_states):
			if i==j:
				model.add_transition(states[i], states[j], not_transitioning_prob)
			else:
				model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))

	# Bake the model
	model.bake()

	# Train the model only on the trials indicated by off_trials
	model.fit(binned_spikes[off_trials, :], algorithm = 'baum-welch', stop_threshold = threshold, max_iterations = max_iterations, edge_inertia = edge_inertia, distribution_inertia = dist_inertia, verbose = False)
	log_prob = [model.log_probability(binned_spikes[i, :]) for i in off_trials]
	log_prob = np.sum(log_prob)

	# Set up things to return the parameters of the model - the state emission dicts and transition matrix
	state_emissions = []
	state_transitions = np.exp(model.dense_transition_matrix())
	for i in range(n_states):
		state_emissions.append(model.states[i].distribution.parameters[0])

	# Get the posterior probability sequence to return
	posterior_proba = np.zeros((binned_spikes.shape[0], binned_spikes.shape[1], n_states))
	for i in range(binned_spikes.shape[0]):
		c, d = model.forward_backward(binned_spikes[i, :])
		posterior_proba[i, :, :] = np.exp(d)

	# Get the json representation of the model - will be needed if we need to reload the model anytime
	model_json = model.to_json()

	return model_json, log_prob, 2*((n_states)**2 + n_states*(n_units + 1)) - 2*log_prob, (np.log(len(off_trials)*binned_spikes.shape[1]))*((n_states)**2 + n_states*(n_units + 1)) - 2*log_prob, state_emissions, state_transitions, posterior_proba
