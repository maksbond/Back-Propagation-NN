import neuralNetwork as NN

if __name__ == "__main__":
    file2 = open("results", "r")

    weight = file2.read()
    a = NN.neauralNetwork(weight)
    a.study()
    a.save()
    #a.check([0.2, 1.1, 0.5])

    file2.close()