from pomegranate import *

# Setting random variable sun
sun = DiscreteDistribution({
    'umbrella': 0.2,
    'no umbrella': 0.8
})

# Setting random variable rain
rain = DiscreteDistribution({
    'umbrella': 0.7,
    'no umbrella': 0.3
})

# setting states
states = [sun, rain]

# Setting transition table
transitions = [[0.8, 0.2],  # Tomorrow's prediction if today is sun
               [0.3, 0.7]]  # Tomorrow's prediction if today is rain

# Starting probabilities
start = [0.5, 0.5]

# Hidden Markow chain
model = HiddenMarkovModel.from_matrix(transitions, states, start,
                                      state_names=['sun', 'rain'])
model.bake()