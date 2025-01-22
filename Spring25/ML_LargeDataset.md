### Scaling Laws

### Cost of operations

### Naive Estimate of Joint Probability
- Assumes independence conditionally $P(c,x) = P(x|c) P(c)$
- Smooth the estimates 
- Classification vs Density Estimation

### Tasks
- Linear-time tasks
  - small things sequentially 
    ```
    map(lambda x:x*2, range(0,20,4))
    ```
- Parallel Tasks
  - Mappers independent
  - faster but a lot cannot do (sequentials)
  - Sorting operations O(n logn)

### Entity Resolution
- Small vocab Counting: lots of memory 
- Large Vocab word count with sorting
  - Streaming -> sort (shuffle sort) -> streaming


### MapReduce
- Work distribution of reducer 
- BashReduce
- Failure handling:
- Debugging
  - careful about shared information
- Abstractions for MapReduce
  - each row 
  - table: 
    - proposed syntax: group by ...
    - join
### Spark
- Data abstraction
  - RDD (resilient distributioned datasets)
    - immutable, can be marked as persistant [.persist()], partitioned and shared, access with short MapReduce pipelines
    - memory cached RDDs [.cache()]
      - if not cache, need to reload the data
  - Operations: 
    - Transformations: lazy
      - Transformations are like 'code
      - recovery from errors, data is stored multiple copies with how they were created; if break, can restore
      - some transformations are cheap mapper process: partitions are the same 
      - some require expensive shuflfe sort: join, groupby
    - Actions: eager, execute and return object
      - use spark interactively; when debug; bottlenecks for production
  
- Logistic Regression example:
  - gradient code: python builds a closure, read-only value
  - gradient done locally on each worker

- Communicate w/o RDDs
  - Broadcast variables
    - e.g. join forces shuffle sort, but if one dataset is much smaller, broadcast this set as variable
  - accumulators: shared counters; don't know the order of increments
  - closures

### MapReduce History
- Ken Church
- Unix sort (mergesort, parallelized, cached): fixed memory
- Counter object:
  - stream all words
  - flush when: (1) new word (2) max buffer
- Stream sort and parallelize
  - sort by partition data appropriately: hash sort key


### Compare words in two corpus
- count words
- bigram
  - (pxy / (px * py))
- pipeline - basic count
- filtering


### Page Rank
- Inlinks
- page hopper: follow random link or jump to random page
- ranks page based on the time page hopper spends
- expect crowd size
- p60
- 