import matplotlib.pyplot as plt
import egraphs

# Set the theme once at the beginning of your script
egraphs.set_epoch_theme()

# Create your figure like you normally would
plt.plot([1, 2, 3], [4, 5.5, 6], label='Pretty line #1')
plt.plot([1, 2, 3], [4, 5, 6], label='Pretty line #2')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.legend()

# Update the layout of the figure after creating it
egraphs.relayout()

plt.show()
