from Neuron import Neuron

neuron = Neuron(3)

print(neuron.weights)
print(neuron.gradients)
print(neuron.deltas)

print(neuron.activate([1, 1, 1]))

print("sum = " + str(neuron.bias))
for weight in neuron.weights:
    print("sum += " + str(weight) + " * " + str(1))
print("sum == " + str(neuron.sum([1, 1, 1])))
print("potential = " + str(neuron.af(neuron.sum([1, 1, 1]))))

print(neuron)