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
- 
