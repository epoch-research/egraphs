import matplotlib.pyplot as plt
import egraphs


def example1():
    # Set the theme once at the beginning of your script
    egraphs.set_epoch_theme()

    # Create your figure like you normally would
    plot()

    # Update the layout of the figure after creating it
    egraphs.relayout()

    plt.show()


def example2():
    # Reset style
    plt.style.use('default')

    # Use a context manager to set the theme only for a specific block of code
    with egraphs.epoch_theme():
        # Create your figure like you normally would
        plot()

        # Update the layout of the figure after creating it
        egraphs.relayout()

        plt.show()

    # New figures won't use the Epoch theme
    plot()
    plt.show()


def plot():
    plt.plot([1, 2, 3], [4, 5.5, 6], label='Pretty line #1')
    plt.plot([1, 2, 3], [4, 5, 6], label='Pretty line #2')
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.legend()


if __name__ == '__main__':
    example1()
    example2()
