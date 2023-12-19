
from textgenrnn import textgenrnn

# textgen = textgenrnn()
# textgen.generate()
textgen = textgenrnn(multi)
textgen.train_from_file('sample_dataset.txt', num_epochs=10)
generated_text = textgen.generate()
print(generated_text)


