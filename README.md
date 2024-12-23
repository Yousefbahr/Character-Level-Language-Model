# Character-Level-Language-Model

![loss_through_time](https://github.com/user-attachments/assets/81d19ffa-80c8-46a3-904d-0f1de6964e48)

TensorFlow implementation of a multi-layer character-level LSTM based on Shakespeare text.

After Training for 50 epochs and with an input statement of 
**"From fairest creatures "**: 

> the sty be blood and love of thee thy self are painting all many, will thee thy silge the world thee

## Architecture
A many-to-one 2-layer LSTM is fed a one-hot encoded input of length equal to the number of unique characters in the document. The last unit of the LSTM is fed into a fully connected output layer, which outputs a probability distribution over the number of unique characters in the document. The input is a sequence of length Tx. The output is a one-hot encoded character.

## Training
The model is trained to predict the next character of a sequence given the preceding Tx characters. This is implemented by partitioning the input X into the shape of (m, Tx, unique_chars), where m is the number of samples; Tx is the length of each sequence; and unique_chars is the length of the unique characters in the document - Y of shape (m, unique_chars).


## Generating A Sequence
To generate a sequence, the model is fed an input from the user of an arbitrary number of characters of length input_Tx. The model predicts the next character by taking the arg max of the probability distribution. The predicted character is added to the input while maintaining a fixed context size of input_Tx and repeat, until reaching the targeted number of generated characters.

## Room For Improvement
- In sequence generation, implementing a fixed context size of Tx (which the model was trained for) instead of input_Tx would ensure optimal performance.
    - If input_Tx << Tx: this could be implemented by padding the input and applying masking.
    - If input_Tx >> Tx: use the most right Tx characters and ignore the rest.

- Sampling a random sample from the probability distribution instead of the arg max would ensure more variation and interesting outputs.






