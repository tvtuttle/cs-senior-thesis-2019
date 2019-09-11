# Graphs scores stored in a specified score file (.m file)
# Be sure that you're opening a score file and not a memory file, they're both .m but only one will work
# Filename is chosen by editing filename variable, no path required

import matplotlib.pyplot as plt
import pickle


# type filename here
filename = "pickled_scores_norm.m"
scores = pickle.load(open("scores/" + filename, "rb"))
print(len(scores))
plt.plot(scores)
plt.show()