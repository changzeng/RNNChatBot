import nltk,json,random,thulac
import numpy as np

thulac = thulac.thulac(seg_only=True)

class RNNNumpy:
	def __init__(self, word_dim=8000, hidden_dim=100, bptt_truncate=4):

		print("Initializing...")
		try:
			fd = open("parameter.data")
			self.load_parameter(fd)
			self.load_from_file = True
			fd.close()
		except:
			# Assign instance variables
			self.word_dim = word_dim
			self.hidden_dim = hidden_dim
			self.bptt_truncate = bptt_truncate
			# Randomly initialize the network parameters
			self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
			self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
			self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
			self.load_from_file = False
			self.num_examples_seen = 0

		self.unknown_token = "UNKNOWN_TOKEN"
		self.sentence_start_token = "SENTENCE_START"
		self.sentence_end_token = "SENTENCE_END"

	def word2index(self,X):
		result = []
		for x in X:
			tmp = []
			for item in x:
				try:
					index_tmp = self.words_index[item]
				except:
					index_tmp = self.words_index[self.unknown_token]

				tmp.append(index_tmp)
			result.append(tmp)

		return result

	# Load data
	def load(self):
		print("Loading data...")
		self.X = []
		self.Y = []

		result = []
		with open("train.txt") as fd:
			for line in fd:
				line = line.strip()
				tmp = line.split("/")
				self.X.append(tmp[:-1])
				self.Y.append(tmp[1:])
				result += tmp

		freq = nltk.FreqDist(result)
		freq_tuple_list = []
		for key in freq:
			freq_tuple_list.append((key,freq[key]))

		freq_tuple_list.sort(key=lambda x:x[1],reverse=True)
		self.words = freq_tuple_list[:self.word_dim+2]
		self.words = [ tmp[0] for tmp in self.words ]

		try:
			self.words.remove(self.sentence_start_token)
		except:
			pass
		try:
			self.words.remove(self.sentence_end_token)
		except:
			pass

		self.words.insert(0,self.unknown_token)
		self.words.insert(0,self.sentence_end_token)
		self.words.insert(0,self.sentence_start_token)

		self.words = self.words[:self.word_dim]

		self.words_index = {}
		index = 0
		for word in self.words:
			self.words_index[word] = index
			index += 1

		self.input_X = self.word2index(self.X)
		self.input_Y = self.word2index(self.Y)

	def softmax(self,x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def forward_propagation(self, x):
		# The total number of time steps
		T = len(x)
		# During forward propagation we save all hidden states in s because need them later.
		# We add one additional element for the initial hidden, which we set to 0
		s = np.zeros((T + 1, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		# The outputs at each time step. Again, we save them for later.
		o = np.zeros((T, self.word_dim))
		# For each time step...
		for t in np.arange(T):
			# Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
			s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
			o[t] = self.softmax(self.V.dot(s[t]))
		return [o, s]

	def predict_next_word(self,x):
		return self.predict(x)[-1]

	def predict_next_prob(self,x):
		o,s = self.forward_propagation(x)
		return o[-1]

	def predict(self,x):
		o,s = self.forward_propagation(x)
		return np.argmax(o,axis=1)

	def calculate_total_loss(self, x, y):
	    L = 0
	    # For each sentence...
	    for i in np.arange(len(y)):
	        o, s = self.forward_propagation(x[i])
	        # We only care about our prediction of the "correct" words
	        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
	        # Add to the loss based on how off we were
	        L += -1 * np.sum(np.log(correct_word_predictions))
	    return L

	def calculate_loss(self, x, y):
	    # Divide the total loss by the number of training examples
	    N = np.sum((len(y_i) for y_i in y))
	    return self.calculate_total_loss(x,y)/N

	def bptt(self, x, y):
	    T = len(y)
	    # Perform forward propagation
	    o, s = self.forward_propagation(x)
	    # We accumulate the gradients in these variables
	    dLdU = np.zeros(self.U.shape)
	    dLdV = np.zeros(self.V.shape)
	    dLdW = np.zeros(self.W.shape)
	    delta_o = o
	    delta_o[np.arange(len(y)), y] -= 1.
	    # For each output backwards...
	    for t in np.arange(T)[::-1]:
	    	dLdV += np.outer(delta_o[t], s[t].T)
	    	# Initial delta calculation
	    	delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
	    	# Backpropagation through time (for at most self.bptt_truncate steps)
	    	for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
	    		# print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
	    		dLdW += np.outer(delta_t, s[bptt_step-1])
	    		dLdU[:,x[bptt_step]] += delta_t
	    		# Update delta for next step
	    		delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
	    return [dLdU, dLdV, dLdW]

	# Performs one step of SGD.
	def numpy_sgd_step(self, x, y, learning_rate):
	    # Calculate the gradients
	    dLdU, dLdV, dLdW = self.bptt(x, y)
	    # Change parameters according to gradients and learning rate
	    self.U -= learning_rate * dLdU
	    self.V -= learning_rate * dLdV
	    self.W -= learning_rate * dLdW

	# Outer SGD Loop
	# - model: The RNN model instance
	# - X_train: The training data set
	# - y_train: The training data labels
	# - learning_rate: Initial learning rate for SGD
	# - nepoch: Number of times to iterate through the complete dataset
	# - evaluate_loss_after: Evaluate the loss after this many epochs
	def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
	    # We keep track of the losses so we can plot them later
	    losses = []
	    for epoch in range(nepoch):
	        # Optionally evaluate the loss
	        # if (epoch % evaluate_loss_after == 0):
	        #     loss = self.calculate_loss(X_train, y_train)
	        #     losses.append((num_examples_seen, loss))
	        #     time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	        #     print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
	        #     # Adjust the learning rate if loss increases
	        #     if (len(losses) == 1 and losses[-1][1] == losses[-2][1]):
	        #         learning_rate = learning_rate * 0.5 
	        #         print("Setting learning rate to %f" % learning_rate)
	        #     sys.stdout.flush()
	        # For each training example...
	        for i in range(len(y_train)):
	            # One SGD step
	            self.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
	            self.num_examples_seen += 1
	            if self.num_examples_seen % 1000 == 0:
	            	print("epoch %d and %d time..." % (epoch,self.num_examples_seen))
	            	print("saving parameter...")
	            	self.save_parameter()

	def random_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
	    # We keep track of the losses so we can plot them later
	    losses = []
	    for epoch in range(nepoch):
	    	for i in range(len(y_train)):
	            # One SGD step

	            num = random.randint(0,len(y_train)-1)

	            self.numpy_sgd_step(X_train[num], y_train[num], learning_rate)
	            self.num_examples_seen += 1
	            if self.num_examples_seen % 100 == 0:
	            	print("epoch %d and %d time..." % (epoch,self.num_examples_seen))
	            	print("saving parameter...")
	            	self.save_parameter()
	            	print("saving data done")
	            	print(self.generate_sentence("你怎么了，不开心吗？"))

	#转换为list
	def to_list(self,array):
		result = []
		for item in list(array):
			result.append(list(item))
		return result

	def save_parameter(self):
		save_dist = {}
		save_dist["U"] = self.to_list(self.U)
		save_dist["V"] = self.to_list(self.V)
		save_dist["W"] = self.to_list(self.W)

		save_dist["word_dim"] = self.word_dim
		save_dist["hidden_dim"] = self.hidden_dim
		save_dist["bptt_truncate"] = self.bptt_truncate

		save_dist["input_X"] = self.to_list(self.input_X)
		save_dist["input_Y"] = self.to_list(self.input_Y)

		save_dist["num_examples_seen"] = self.num_examples_seen

		save_dist["words"] = self.words
		save_dist["words_index"] = self.words_index

		with open("parameter.data","w") as fd:
			json.dump(save_dist,fd)

	def load_parameter(self,fd):
		save_dist = json.load(fd)
		self.U = np.array(save_dist["U"])
		self.V = np.array(save_dist["V"])
		self.W = np.array(save_dist["W"])

		self.word_dim = save_dist["word_dim"]
		self.hidden_dim = save_dist["hidden_dim"]
		self.bptt_truncate = save_dist["bptt_truncate"]

		self.input_X = np.array(save_dist["input_X"])
		self.input_Y = np.array(save_dist["input_Y"])

		self.num_examples_seen = save_dist["num_examples_seen"]

		self.words = save_dist["words"]
		self.words_index = save_dist["words_index"]

	def train(self):
		if not self.load_from_file:
			self.load()

		print("Start to training...")
		self.train_with_sgd(self.input_X,self.input_Y,nepoch=200)

	def random_train(self):
		if not self.load_from_file:
			self.load()

		print("Start to training...")
		self.random_sgd(self.input_X,self.input_Y,nepoch=200)

	def generate_sentence(self,inputs):
		inputs = [ token[0] for token in thulac.cut(inputs) ]
		inputs = [self.sentence_start_token] + inputs + [self.sentence_end_token]
		inputs = self.word2index(inputs)

		# We start the sentence with the start token
		new_sentence = inputs
		# Repeat until we get an end token
		while not new_sentence[-1] == self.words_index[self.unknown_token]:
			next_word_probs = self.predict_next_prob(new_sentence)
			sampled_word = self.words_index[unknown_token]
			# We don't want to sample unknown words
			while sampled_word == self.words_index[unknown_token]:
				samples = np.random.multinomial(1, next_word_probs[-1])
				sampled_word = np.argmax(samples)

			if sampled_word == self.words_index[self.sentence_end_token]:
				break

			new_sentence.append(sampled_word)

		sentence_str = [self.words[x] for x in new_sentence[1:]]
		sentence_str = "".join(sentence_str)
		sentence_str = sentence.split(self.sentence_end_token)

		return sentence_str[1]
	 
rnn = RNNNumpy()
rnn.random_train()