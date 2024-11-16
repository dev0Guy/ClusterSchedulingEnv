# Cluster Environment Gym

The `ClusterEnv` environment is a simulation of a cluster system designed to manage the scheduling and allocation of jobs 
on machines with limited resources. It is built using the `gymnasium` library and supports custom configurations for jobs, machines, and resources.


## Wrappers

### Observation Wrapper

#### CombineMachinJobWrapper
The wrapper combines two separate observation components, jobs and machines, into a single unified observation space. 

### Action Wrapper

#### DiscreteActionWrapper
The wrapper simplifies the action space by converting it into a discrete set of actions, 
enabling agents to operate within a simpler action space. Here’s a detailed breakdown of its functionality:
In the ClusterEnv, the default action space may involve complex multi-dimensional actions, such as specifying machine-job pairings or time management commands. This wrapper:
1.	Converts the action space into a discrete space of size  `(J * M + 1)`: 
     - additional action represents a “skip time” operation.
     - The remaining actions represent all possible combinations of jobs and machines.
2.	Transforms a discrete action into a corresponding multi-dimensional action understandable by the ClusterEnv
### Reward Wrapper
#### TODO:

### Env Wrapper

#### ScheduleFromSelectedTimeWrapper
The wrapper customizes job scheduling in the ClusterEnv by dynamically validating and scheduling jobs based on resource availability
and arrival times. It ensures jobs are scheduled only if sufficient resources are available on the target machine for 
the required duration using a sliding window approach. The wrapper updates machine resource states upon successful scheduling, preventing
over-allocation or invalid assignments. This ensures efficient resource utilization and supports dynamic scheduling in resource-constrained environments.

#### QueueWrapper
The wrapper modifies the ClusterEnv to focus on a subset of jobs (queue_size), sorting them by status and dynamically updating their order based 
on priority (e.g., pending jobs). It adjusts the action space to reference only the selected subset of jobs and remaps 
job indices in actions accordingly. The observation space is modified to include only the top queue_size jobs and their statuses, ensuring that the agent operates on the most relevant subset.
This wrapper streamlines decision-making by reducing complexity while prioritizing pending jobs for scheduling.


## Extra Information

### Notations:
-   **J** : Number of jobs
-   **R:** Number of resources
-   **T:** Number of ticks
-   **A:** Job arrival rate 

### Observation space
The environment provides the following observation:
-   Machines: Free space per resource over time. Shape: `(M, R, T)`
-   Jobs: Resource utilization demands over time. Shape: `(J, R, T)`
-   Status: Current state of each job `(Not Created, Pending, Running, Completed)`. Shape: `(J)`

### Action space
consist of gym.spaces.MultiDiscrete of `[TickActionOptions, M, J]` </br>
where `TickActionOptions`: </br>
`0` - No time increment </br>
`1` - Increment time

### Job Generation Process

**The following describes the mathematical process used to generate jobs**
1. **Job Arrival Times -** </br>
Job arrival times are modeled as a Bernoulli process:
```math
T_{\text{enter}, i} = \sum_{j=1}^{i} B_j, \quad B_j \sim \text{Bernoulli}(p)
```

2. **Job Durations -** </br>
Job durations are sampled from a mixture of two uniform distributions:
```math
D_i = 
\begin{cases} 
\text{Uniform}(1, 3), & \text{if } U_i < 0.8 \\
\text{Uniform}(10, 15), & \text{otherwise}
\end{cases}
\quad \text{where} \quad U_i \sim \text{Uniform}(0, 1)
```

3.  **Resource Demands -** </br>
Each job (i) is assigned resource demands based on its dominant resource:
```math
R_{i,k} =
\begin{cases} 
\text{Uniform}(127.5, 255), & \text{if } k = R_{\text{dom}, i} \\
\text{Uniform}(25.5, 51), & \text{if } k \neq R_{\text{dom}, i}
\end{cases}
\quad \text{where} \quad R_{\text{dom}, i} \sim \text{Uniform}\{0, \text{R}-1\}
```

4. **Job Activity Over Time -** </br>
Jobs are active based on their arrival times and durations:
```math
A_{i,t} =
\begin{cases}
1, & \text{if } T_{\text{enter}, i} \leq t < T_{\text{enter}, i} + D_i \\
0, & \text{otherwise}
\end{cases}
```

5. **Final Resource Utilization-** </br>
The 3D array representing job resource utilization is calculated as:
```math
J_{i,k,t} = A_{i,t} \cdot R_{i,k}
J_{i,k,t} = A_{i,t} \cdot R_{i,k}
```

### Machines Generation Process
Each machine is reset with 255.0 of shape `(J,R,T)`
