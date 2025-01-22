### venv and pip
- To check version of package: 
    ```
    pip show pandas
    ```


    
### Pandas
- To filter pandas with a list 
    ```python
    df = df[~df.col.isin(to_remove)]
    ```
- Filter pandas with multiple conditions
  ```python
  df.loc[(condition1) & (condition2)]
  ```
- To make series into bins: pd.cut(series, bins=n)
- Set each element in the list in a column: explode
- Lambda functions 
  - Syntax: lambda arguments : expression
    - Also called anonymous function
    - Syntax sugar to define a function
  - groupby
  ```python
  # to filter 
  dfg = df[['key', 'value']].groupby(['key'])
  dfg = dfg.filter(lambda x:x['value'].count()>threshold)

  ```
### Lists
- To sort a list based on values in another:
  ```
  [x for _, x in sorted(zip(reference_lst, list_to_srt))]
  ```


### regex
- raw string r'abc\t'
- match
  ```python
  import re
  print(r'raw \t string')
  pattern = re.compile(r'abc')
  matches = pattern.findter(text_to_search)
  for match in matches:
    print(match)
  ```
- Meta Characters:  . ^ $ * + ? { } [ ] \ | ( )
  - . : dot matches any character except for new line \n
  - \d Digit (0-9)
  - \D Not a digit (capital case = negation)
  - \w word character (a-z, A-Z, 0-9, _)
  - \s Whitespace (space, tab, newline)
- Anchors
  - \b word boundary (white space or non alphanumeric): finds patterns that is the start of the word or have whitespace before them 
  - ^: beginning of a string
  - $: End of string
  - *: 0 or more chars
  - ?: have 0 or 1



### hypothesis testing
- null hypothesis $H_0$, alternative hypothesis
  - $H_0$ refers something we wish to reject
  - $H_A$ refers to demonstrate as more possible than null
- confidence threshold $\alpha$ - minimum to reject the null
- t-test
- friedman test
- error
  - type I - false positive -> reject null when should be accepted
  - type II - false negative -> accept null when should be rejected
- p-value

### Data wrangling
- missing values
  - missing completely at random 
  - missing at random - probability missing is the same within certain groups
  - not missing at random - probability of data being missing varies for reasons unknown
- Imputation
  - mean value
  - hot and cold deck imputation
  - regression 
  - interpolation
  - ommission
    - pairwise deletion - just the available values
    - listwise deletion - remove data that has one or more missing values

### Feature engineering
- imputation
- binning 
- outliers
- log transform
- one-hot encoding
- binary feature encoding
- ordinal feature encoding
- feature split - easier to bin, improves model performance
- scaling
- gradient descent
- PCA
  - ensure independence of variables
  - linear transformation technique
  - eigen decomposition on covariance matrix
  - unsupervised
- t-SNE 
  - visualization of clusters of points in higher dimensions
  - define probability of some data i picking another point j
  - KL divergence
  - far better visualization than PCA
  - optimization -> larger datasets require larger perlexity but too large can remove potential clusters
  - stochastic -> distance between points mean little
  - sensitive to data scaling
  - non-convex loss function -> initilization matters

### deep learning
- ANN: multi-layer perceptron
- CNN: early layers recognize features later layers combine features
  - receptive fields feeds intro convolutional layer
  - pooling: pytorch output_size = (input_size - kernel_size + 2 * padding) / stride + 1
- RNN: connect feed back to prior layers
  - back-propagation in time
  - vanishing and exploding gradient problems
- LSTM: 
  - memory cell : retain value for a short or long time as a function of inputs
  - 3 gates 
    - input gate: new information into memory 
    - forget gate: existing piece of information forgotten to remmeber new data
    - output gate: when information used in output
    - cell contains weights controling each gate
    - BPTT - optimization
  - GRU: gated recurrent unit networks
    - simiplication of LSTM 
    - 2 gates: no output gate
      - update gate: how mych previous cell contents to maintain
      - reset gate: how to incorporate new input with previous cell contents
    - standard RNN: reset gate =1, update gate=0
    - 


### NLP
- Analytics, generalization, acquisition
- TF-IDF = TF (term frequency) x IDF (inverse document frequency) 
  - high frequency words have low idf
  - suffer curse of dimensionality
  - no context or word order
- word embeddings
  - Word2Vec: shallow feedforward nn and leverage occurrence within local context
  - GloVe matrix factorization and leverage glocal word to word occurrence counts
  - both are similar
  - Do not handle polysemy well -> combine semantic representations of different senses of a word into one vector
- ELMo solve ^^ problem 
  - Bi-LSTM -> look at entire sentence before assigning embedding
  - predict next word
  - Same general word embeddings are not enough in all kinds of NLP tasks -> need fine-tuning
- ULM-FiT -> inductive transfer learning method
  - general domean LM pre-training: general features of language
  - target task discriminative fine-tuning: changing learning rate
  - target task classifier fine-tuning: gradual unfreezing and repeating stage 2; preserve low-level representations and adapt to high-level ones
- Transformer
- Traditional Seq2Seq models
  - usually encodre-decoder architectore; both are RNN with LSTM or GRU units
  - enoder: input sequence to context vector
  - decoder: generate result one-word embedding at a time
- Attention
- Transformers
  - self-attention/ intra-attention
    - encoder: richer representations of input in parallel
    - decoder: no hidden states self-attention on all outputs till that point
  - encoder: 
    - 