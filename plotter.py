
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Plotter(object):

    def __init__(self):
        self.fig = plt.figure(figsize=(4, 4))
        plt.ion()
        plt.show()

    def plot(self, samples):
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        self.fig.canvas.draw()
