import numpy as np
import multiprocessing as mp

def logp(data, p, states):
	return np.sum(np.log(p[states.astype('int'), np.tile(data.reshape(data.shape[0], 1, data.shape[1]), (1, states.shape[1], 1)).astype('int')]), axis = -1)

def E_step(data, identity, palatability, switchlim1, switchlim2, p):
	#switchpoints = np.array([[i, j] for i in range(switchlim1[0], switchlim1[1], 1) for j in range(i + switchlim2[0], switchlim2[1], 1)])
	#states = find_states(identity, palatability, switchpoints, data.shape[0])
	#for switchpoint1 in range(switchlim1[0], switchlim1[1], 1):
	#	for switchpoint2 in range(switchpoint1 + switchlim2[0], switchlim2[1], 1):
	#		states = find_states(identity, palatability, [switchpoint1, switchpoint2], data.shape[0])
	#		loglik_list.append([switchpoint1, switchpoint2, logp(data, p, states)])

	loglik_list = logp(data, p, states)
	max_loglik = np.argmax(loglik_list)
	return loglik_list[max_loglik], switchpoints[max_loglik, :]

def find_states(identity, palatability, switchpoints, data):
	#states1 = np.where(np.arange(length) <= switchpoints[0], np.zeros(length), identity*np.ones(length))
	#states = np.where(np.arange(length) <= switchpoints[1], states1, palatability*np.ones(length))
	#states1 = np.where(np.tile(np.arange(length).reshape(1, length), (switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 0].reshape(switchpoints.shape[0], 1), (1, length)), 0, identity)
	#states = np.where(np.tile(np.arange(length).reshape(1, length), (switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 1].reshape(switchpoints.shape[0], 1), (1, length)), states1, palatability)
	states1 = np.where(np.tile(np.arange(data.shape[1]).reshape(1, 1, data.shape[1]), (data.shape[0], switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 0].reshape(1, switchpoints.shape[0], 1), (data.shape[0], 1, data.shape[1])), np.zeros((data.shape[0], switchpoints.shape[0], data.shape[1])), np.tile(identity.reshape(data.shape[0], 1, 1), (1, switchpoints.shape[0], data.shape[1])))
	states = np.where(np.tile(np.arange(data.shape[1]).reshape(1, 1, data.shape[1]), (data.shape[0], switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 1].reshape(1, switchpoints.shape[0], 1), (data.shape[0], 1, data.shape[1])), states1, np.tile(palatability.reshape(data.shape[0], 1, 1), (1, switchpoints.shape[0], data.shape[1])))
	return states

def normalize_p(p):
	return p/np.tile(np.sum(p, axis = 1).reshape((p.shape[0], 1)), (1, p.shape[1]))

def fit(data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions, restart):

	np.random.seed(restart)

	identity = identity.astype('int')
	palatability = palatability.astype('int')
	p = np.random.random((num_states, num_emissions))
	p = normalize_p(p)
	switches = []

	switchpoints = np.array([[i, j] for i in range(switchlim1[0], switchlim1[1], 1) for j in range(i + switchlim2[0], switchlim2[1], 1)])
	states = find_states(identity, palatability, switchpoints, data)
	logp_list = []
	loglik_list = []
	exp_loglik_list = []
	converged = 0
	for i in range(iterations):
		switches = []
		loglik_list = logp(data, p, states)

		# Initially, we used the max/mode of the probability of the switchpoints given the p and the data
		# Now we will work with the mean of that probability instead
		max_loglik = np.argmax(loglik_list, axis = 1)
		logp_list.append(np.sum(np.max(loglik_list, axis = 1)))
		switches = switchpoints[max_loglik, :]
		max_states = states[np.arange(data.shape[0]), max_loglik, :]

		# loglik_list is shaped # trials x # of putative switchpoints. We first exponentiate the logprobs to probs, then scale/normalize across the putative switchpoints to get the posterior probability of the switchpoints given the data and p
		exp_loglik_list = np.exp(loglik_list)
		exp_loglik_list /= np.tile(np.sum(exp_loglik_list, axis = 1).reshape(loglik_list.shape[0], 1), (1, loglik_list.shape[1]))
		# Find the mean switchpoints for every trial by multiplying the array of switchpoints by their posterior probabilities in exp_loglik_list - recast to integers
#		mean_switches = np.sum(np.tile(exp_loglik_list.reshape(loglik_list.shape[0], loglik_list.shape[1], 1), (1, 1, 2))*np.tile(switchpoints.reshape(1, switchpoints.shape[0], switchpoints.shape[1]), (loglik_list.shape[0], 1, 1)), axis = 1).astype('int')

		# Now find the state sequence given by these mean switchpoints
		# We can use the find_states func, but with a twist. The find_states func first finds the state sequence determined by each input pair of switchpoints for every trial
		# So it gives us an array of shape # trials x # switchpoints x time
#		mean_states = find_states(identity, palatability, mean_switches, data)
		# But we know that the switchpoints we are feeding in here are specific to trials - so switchpoint 1 corresponds to trial 1 and so on. So we associate the switchpoints that way and get an array of shape # trials X time
#		mean_states = np.array([mean_states[trial, trial, :] for trial in range(mean_states.shape[0])])
		# Now we get the log likelihood of the data with these mean switchpoints using the logp function
		# We need to get mean_states into a comparable shape as the states array which is usually fed into logp. So we add an axis in the middle of mean_states
#		logp_list.append(np.sum(logp(data, p, mean_states.reshape(mean_states.shape[0], 1, mean_states.shape[1]))))

#		for trial in range(data.shape[0]):
#			logp_max, switches_max = E_step(data[trial], identity[trial], palatability[trial], switchlim1, switchlim2, p)
#			states = find_states(identity[trial], palatability[trial], switchpoints, data.shape[1])
#			loglik_list = logp(data[trial], p, states[trial])
#			max_loglik = np.argmax(loglik_list)
#			this_logp += loglik_list[max_loglik]
#			switches.append(switchpoints[max_loglik, :])

#		switches = np.array(switches).astype('int')
#		logp_list.append(this_logp)


		p_numer = np.zeros((num_states, num_emissions))
		# Concatenate the logp maximizing state sequence and data together, and then find the counts of the (state, emission) pairs
		unique_pairs, unique_counts = np.unique(np.vstack((max_states.flatten(), data.flatten())), axis = 1, return_counts = True)
		unique_pairs = unique_pairs.astype('int')
		# Now add the unique counts to the right (state, emission) pair
		p_numer[unique_pairs[0, :], unique_pairs[1, :]] += unique_counts
		# Normalizing these counts of emissions in every state will directly give the right p
		# Add a small number to the counts in case one of them is 0 - in that case, calculating logs gives "DivideByZeroError"
		p = normalize_p(p_numer + 1e-14)

#		p_numer = np.zeros((num_states, num_emissions))
#		p_denom = np.zeros((num_states, num_emissions))
#		for trial in range(data.shape[0]):
#			for emission in range(num_emissions):
#				p_numer[0, emission] += np.sum(data[trial][:switches[trial][0]] == emission)
#				p_denom[0, emission] += switches[trial][0]
#				p_numer[identity[trial], emission] += np.sum(data[trial][switches[trial][0]:switches[trial][1]] == emission)
#				p_denom[identity[trial], emission] += switches[trial][1] - switches[trial][0]
#				p_numer[palatability[trial], emission] += np.sum(data[trial][switches[trial][1]:] == emission)
#				p_denom[palatability[trial], emission] += data[trial].shape[0] - switches[trial][1]
#		p = p_numer/p_denom
#		Add a small number to the probabilities in case one of them is 0 - in that case, calculating logs gives "DivideByZeroError"
#		p = normalize_p(p + 1e-14)

		if i > 1 and np.abs(logp_list[-1] - logp_list[-2]) < threshold:
			converged = 1
			break

	return logp_list, p, switches, converged, exp_loglik_list, switchpoints

def implement_EM(restarts, n_cpu, data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions):
	pool = mp.Pool(processes = n_cpu)

	results = [pool.apply_async(fit, args = (data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions, restart,)) for restart in range(restarts[0], restarts[1])]
	output = [result.get() for result in results]

	converged_seeds = np.array([i for i in range(len(output)) if output[i][3] == 1])
	if len(converged_seeds) == 0:
		print("Another round of {:d} seeds running as none converged the first time round".format(restarts[1]))
		rand_int = np.random.randint(1, 10)
		implement_EM((rand_int*restarts[1], (rand_int + 1)*restarts[1]), n_cpu, data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions)
	else:
		logprobs = np.array([output[i][0][-1] for i in range(len(output))])
		max_logprob = np.argmax(logprobs[converged_seeds])
		return output[converged_seeds[max_logprob]][0], output[converged_seeds[max_logprob]][1], output[converged_seeds[max_logprob]][2], output[converged_seeds[max_logprob]][4], output[converged_seeds[max_logprob]][5]
