import numpy as np
from enum import Enum

class Observed(Enum):
    Surf = 0
    Beach = 1
    Videogame = 2
    Study = 3
    Unknown = 4

class Hidden(Enum):
    Sunny = 0
    Windy = 1
    Rainy = 2

def load_dataset():
    start = Hidden.Rainy
    stop = Hidden.Sunny
    stop_unknown = None

    emissions = np.array([
        # Sunny Windy Rainy
        [0.4, 0.5, 0.1],  # Surf
        [0.4, 0.1, 0.1],  # Beach
        [0.1, 0.2, 0.3],  # Videogame
        [0.1, 0.2, 0.5],  # Study
        [1./3, 1./3, 1./3]])  # Unknown

    transitions = np.array([
        # Sunny Windy Rainy
        [0.7, 0.3, 0.2],  # Sunny
        [0.2, 0.5, 0.3],  # Windy
        [0.1, 0.2, 0.5]])  # Rainy

    history_observed = np.array([Observed.Videogame,
                                 Observed.Study,
                                 Observed.Study,
                                 Observed.Surf,
                                 Observed.Beach,
                                 Observed.Videogame,
                                 Observed.Beach])

    history_unknown = np.array([Observed.Unknown,
                                 Observed.Unknown,
                                 Observed.Unknown,
                                 Observed.Unknown,
                                 Observed.Unknown,
                                 Observed.Unknown,
                                 Observed.Unknown])

    assert emissions.sum(axis=0).all()
    assert transitions.sum(axis=0).all()

    return emissions, transitions, history_observed, history_unknown, start, stop, stop_unknown

def viterbi(emissions, transitions, history_observed, start, stop):
    V = np.zeros((len(history_observed),len(transitions)), dtype=float)
    phi = np.zeros((len(history_observed),len(transitions)), dtype=int)

    # Initialize forward pass
    V[0] = np.log(transitions[:,start.value]) + np.log(emissions[history_observed[0].value,:])

    # Forward pass: incrementally fill the table
    for i in range(1,len(history_observed)):
        v = np.zeros((len(transitions), len(transitions)))
        for j in range(len(transitions)):
            for k in range(len(transitions)):
                v[j,k] = np.log(transitions[j, k]) + np.log(emissions[history_observed[i].value, j]) + V[i-1, k]

            V[i,j] = np.max(v[j,:])
            phi[i,j] = np.argmax(v[j,:])

    # Initialize backward pass
    y = np.zeros(len(transitions))
    for j in range(len(transitions)):
        y[j] = np.log(transitions[stop.value,j]) + V[len(history_observed)-1, j]

    y_hat = np.zeros(len(history_observed))
    y_hat[-1] = np.argmax(y)

    # Backward pass: follow backpointers
    for i in range(len(history_observed)-2, -1, -1):
        y_hat[i] = phi[i+1,int(y_hat[i+1])]

    return y_hat

def forward_backward(emissions, transitions, history_observed, start, stop):
    alpha = np.zeros((len(history_observed), len(transitions)), dtype=float)
    beta = alpha.copy()

    # Initialize forward pass
    alpha[0] = transitions[:,start.value] * emissions[history_observed[0].value, :]

    # Forward pass: incrementally fill the table
    for i in range(1, len(history_observed)):
        v = np.zeros((len(transitions), len(transitions)))
        for j in range(len(transitions)):
            for k in range(len(transitions)):
                v[j, k] = transitions[j, k] * emissions[history_observed[i].value, j] * alpha[i - 1, k]

            alpha[i, j] = np.sum(v[j, :])

    # Initialize backward pass
    if stop:
        beta[-1, :] = transitions[stop.value, :]
    else:
        beta[-1, :] = 1./len(transitions[0])

    # Backward pass: compute backward probabilities
    for i in range(len(history_observed)-2, -1, -1):
        v = np.zeros((len(transitions), len(transitions)))
        for j in range(len(transitions)):
            for k in range(len(transitions)):
                v[j, k] = transitions[k, j] * emissions[history_observed[i+1].value, k] * beta[i + 1, k]

            beta[i, j] = np.sum(v[j, :])

    post_unigram = alpha * beta

    likelihood = np.zeros(len(history_observed), dtype=float)
    for i in range(len(history_observed)):
        likelihood[i] = np.sum(alpha[i, :] * beta[i, :])
        post_unigram[i] /= likelihood[i]

    y_hat = np.zeros(len(history_observed))
    for i in range(len(history_observed)):
        y_hat[i] = np.argmax(post_unigram[i])

    return post_unigram, y_hat

def profit(probs, bet):
    expected_profit = np.zeros(len(probs[:,0]))
    for i in range(len(probs[:,0])):
        m = np.max(probs[i, :] * bet)
        expected_profit[i] = m * bet - (1-m) * bet

    return np.sum(expected_profit)

def main():
    emissions, transitions, history_observed, history_unknown, start, stop, stop_unknown = load_dataset()
    # print('Emissions P(x|y) = [x,y] = ')
    # print(emissions)
    # print('Transitions P(y(t+1)|y(t)) = [y(t+1),y(t)]')
    # print(transitions)

    print('Question 2 - 1a)')
    y_hat = viterbi(emissions, transitions, history_observed, start, stop)

    most_likely_weather = []
    for i in range(len(y_hat)):
        most_likely_weather.append(Hidden(y_hat[i]).name)

    print('Viterbi Most Likely Weather : ',most_likely_weather)

    prob, y_hat = forward_backward(emissions, transitions, history_observed, start, stop)

    most_likely_weather = []
    for i in range(len(y_hat)):
        most_likely_weather.append(Hidden(y_hat[i]).name)

    print('Question 2 - 1b)b)')

    expected_profit = profit(prob, 1.0)
    print('After knowing Johns activities : ', round(expected_profit,2))
    print('Most Likely Weather : ', most_likely_weather)

    print('Question 2 - 1b)a)')
    prob, y_hat = forward_backward(emissions, transitions, history_unknown, start, stop_unknown)

    expected_profit = profit(prob, 1.0)

    print('Before knowing Johns activities : ', round(expected_profit,2))

    most_likely_weather = []
    for i in range(len(y_hat)):
        most_likely_weather.append(Hidden(y_hat[i]).name)

    print('Most Likely Weather : ', most_likely_weather)

if __name__ == "__main__":
    main()
