import matplotlib.pyplot as plt
scores = []
with open("log.txt", "r") as file: 
    line = file.read().splitlines()
scores =[int(l) for l in line]
epochs = [epoch for epoch in range (300000) if epoch % 1000 == 0]
lines = plt.plot(epochs, scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.savefig("pic.png")