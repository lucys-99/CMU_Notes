### AWS Elasticity 
#### Goal 
- Create scripts that enable horizontal scaling and autoscaling with AWS EC2
- Horizontal scaling: need to reach max rps 50 in 30 min
- Auto Scaling: need to reach max rps and average rps within the limited resource time
#### Horizontal scaling
- Practices creating and deleting: security groups, instances
- Get information about vpc, security groups, instances
- Basic ideas: create a request to be sent with the corresponding client and then wait for the response
- Most tricky part: be aware of components to be returned to the main function for later reference. Sometimes, it is more convenient to return dns and other times, it might be the name or instanceId. 

#### Auto Scaling
- Basic concepts: autoscaling group, elastic load balancer, target group, cloudwatch alarms
- How things connect:
  - Elastic load balancer redirects the traffic to target group
  - Autoscaling group controls how many instances/resources in the target group
  - To making autoscaling group automatic, policies needs to be attached to the group. The metrics are defined in cloudwatch alarms
  - For coding simplicity and structure, it is helpful to separate autoscale, ec2, elb, and cloudwatch parts as separated classes
- Helpful:
  - Read the documentation about each parameters in each request
  - When creating resources, sometimes the api will check its existence. Sometimes, they don't perform checks and return with error. It is always safe to check if resource exists or not. 

#### Web containers
- Docker: 
  - basic idea is to use docker to containerize the code and environment
  - The project helps with getting familiar with the structure of docker code, how to refer path, and push to cloud repositories. 
  - It was helpful to look at the structure of the springboot code before implemetation of dockerfile.
- Kubernetes: 
  - The basic idea is to use kubernetes to maintain the cluster that pulls the image from cloud repo and host containers as pods. 
  - When unexpected behaviors happened, it was important to check the log messages from the pods themselves. Also sometimes the containers in the repo may be outdated. 
- Github CICD:
  - In this part of the project, I practiced using github actions to authorize cloud access, build docker, push to repo and start the deployment on github. 
  - The hardest part is to match the credentials in the action.yaml file for the action to read the passwords successfully. 
  - Tool act helps with debugging the basic CI/CD functionalities locally


#### Apache Spark

- **Distributed Big-Data Processing Framework**
  - Enables scalable, in-memory data processing across clusters.
  - Uses Directed Acyclic Graph (DAG) execution model for optimization.

- Transformations vs. Actions

  - Transformations (Lazy):
    - Return a new RDD/DataFrame and do **not** trigger computation until an action is called.
    - Examples:
      - `map`, `filter`, `flatMap`, `groupBy`, `join`, `select`, `withColumn`, `drop`
    - Laziness allows Spark to optimize the execution plan.

  - Actions (Eager):
    - Trigger actual execution of the DAG and return results or write to storage.
    - Examples:
      - `collect()`, `count()`, `first()`, `take(n)`, `show()`, `write()`, `saveAsTextFile()`, `foreach()`


- RDD vs. DataFrame vs. Dataset
   - RDD (Resilient Distributed Dataset):
     - Low-level API, untyped/unstructured data.
     - Fine-grained control, good for complex data manipulation.
     - Functional transformations (`map`, `reduceByKey`, etc.)
     - No built-in optimization (no Catalyst or Tungsten).

   - DataFrame:
     - Distributed collection of data organized into named columns (like a table in RDBMS).
     - High-level API with built-in optimizations via **Catalyst** (query optimizer).
     - Can use SQL-like operations (`select`, `groupBy`, `filter`, etc.)
     - Backed by RDDs, but optimized.

   - Dataset (only in Scala and Java):
     - Combines the benefits of RDD (type-safety) and DataFrame (performance).
     - Allows compile-time type checking and custom object mapping.

   - All three are **immutable** – any transformation returns a new instance.

- Deployment & Development

  - Local Deployment:
    - Useful for testing and debugging.
    - Tools: `spark-shell`, `PySpark`, notebooks like **Zeppelin**, **Jupyter**, or **Databricks**.

  - Cluster Deployment:
    - Modes: Standalone, YARN, Mesos, Kubernetes.
    - Scheduler distributes tasks across worker nodes.

- Caching and Persistence
  - `cache()`: Persists data **in memory** only (default storage level: `MEMORY_AND_DISK`).
  - `persist()`: Offers multiple storage levels (e.g., memory only, disk only, memory and disk).
    - Use when dataset doesn’t fit entirely in memory.
  - Best Practices:
    - Cache or persist DataFrames/RDDs if reused in multiple actions to avoid recomputation.
    - Call caching **before** the first action that triggers computation.

- Partitioning, Randomness & Ranking Issues
  - Spark distributes data across partitions; operations like `rank`, `dense_rank`, or `row_number` may produce inconsistent results if:
    - Partitioning is not controlled.
    - No deterministic ordering is specified in `orderBy`.

  - **To Ensure Consistency**:
    - Explicitly define `orderBy` before using ranking/window functions.
    - Use `.repartition()` or `.coalesce()` wisely to control data distribution.
    - Random behavior can be more noticeable with large datasets and improper partitioning.

- Extra Notes
  - For scala: documentation for Spark in Scala sucks... Look up pySpark syntax and search for corresponding Scala syntax...
  - Avoid using `collect()` on large datasets – it brings all data to the driver and can cause OOM (Out of Memory) errors.
  - Use broadcast joins to optimize joins with small dimension tables.
  - Use `.explain()` to understand and optimize the physical plan.
  - SparkUI is a helpful way to monitor memory usage. 


#### Agentic workflow

#### Team project
- Backend: 
  - AWS ElastiCache Redis (in memory database) is charged by the storage used. Each cluster has 1 free backup to restore from. 
  - There are 2 types of redis database clusters available: single-node and multi-node [AWS documentation](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/Clusters.html). 
    - Single-node contains all data on the same cache node, which allows replicas. 
    - Multi-node has cluster-mode enabled, providing fault-tolerance and sharding. 
    - To access the nodes, it is safer to deploy Redis databases on the same VPC with EC2 instances. 
- Frontend: 
  - Vert.x 
- Micro service communications: 
  - grpc (grpc-protobuf)
  - api
- AWS EKS clusters