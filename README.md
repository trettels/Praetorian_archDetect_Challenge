Paretorian Machine Learning-based architecture detection challenge

First pass at solving the challenge located at: (https://www.praetorian.com/challenges/machine-learning-challenge/)

My goal here was to try the challenge using something other than a clustering / SVM algorithm.  Ultimately I would want to transition the model from detection to decompilation, either as a summary or as a full decompilation. Currently the model gets a roughly 60-70% success rate on test data, which is too low.  The next steps I will try, given the time, include:

  1) instead of a hard cutoff of bytes based on appearance frequency, reduce the byte dictionary size based on how informative the byte is to the architecture
  2) Switch to an LSTM or related architecture that looks at the bytes serially, rather than as a vector
