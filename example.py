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

    # Reset the style
    plt.style.use('default')


def example2():
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


def example_braces():
    with egraphs.epoch_theme():
        plot()

        egraphs.relayout()

        # Add braces only after relayout
        egraphs.add_brace(plt.gca(), 1, 4, 6, transform=plt.gca().transData)

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
    example_braces()
