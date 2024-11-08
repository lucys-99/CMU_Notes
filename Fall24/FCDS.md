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
