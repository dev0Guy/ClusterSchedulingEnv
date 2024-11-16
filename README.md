# Cluster Environment Gym

The `ClusterEnv` environment is a simulation of a cluster system designed to manage the scheduling and allocation of jobs 
on machines with limited resources. It is built using the `gymnasium` library and supports custom configurations for jobs, machines, and resources.
---
### Notations:
-   **J** : Number of jobs ( n\_jobs )
-   **R:** Number of resources ( n\_resources )
-   **T:** Number of ticks ( n\_ticks )
-   **A:** Job arrival rate ( \_job\_arrival\_rate )
-   **jobs\_array[j, r, t]:** Utilization of resource  r  by job  j  at time  t 
-   **enter\_times[j]:** Time when job  j  arrives
-   **duration[j]:** Duration of job  j 
-   **dominant\_resource[j]:** Dominant resource for job  j 
-   **res\_demand[j, r]:** Demand of resource  r  for job  j 
-   **mask[j, t]:** Boolean mask indicating whether job  j  is active at time  t 

---
## Cluster Base (v0)

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
The following describes the mathematical process used to generate jobs:
1. Job Arrival Times - Job arrival times are modeled as a Bernoulli process:
```math
T_{\text{enter}, i} = \sum_{j=1}^{i} B_j, \quad B_j \sim \text{Bernoulli}(p)
```
where:
- ``` (T_{\text{enter}, i})``` is the arrival time of the (i)-th job.
- ```(B_j)``` represents the outcome of the Bernoulli process ((1) if a job arrives, (0) otherwise).
- ```(p)``` is the job arrival rate.

2. Job Durations - Job durations are sampled from a mixture of two uniform distributions:
```math
D_i = 
\begin{cases} 
\text{Uniform}(1, 3), & \text{if } U_i < 0.8 \\
\text{Uniform}(10, 15), & \text{otherwise}
\end{cases}
```
where:
- ```(D_i)``` is the duration of the (i)-th job.
- ```(U_i \sim \text{Uniform}(0, 1))``` determines if a job is short (80%) or long (20%).
3. Resource Demands - Each job (i) is assigned resource demands based on its dominant resource:
```math
R_{i,k} =
\begin{cases} 
\text{Uniform}(127.5, 255), & \text{if } k = R_{\text{dom}, i} \\
\text{Uniform}(25.5, 51), & \text{if } k \neq R_{\text{dom}, i}
\end{cases}
```
where:
- `(R_{i,k})` is the resource demand of job (i) for resource (k).
- `(R_{\text{dom}, i})` is the dominant resource index for job (i).

4. Job Activity Over Time -Jobs are active based on their arrival times and durations:
```math
A_{i,t} =
\begin{cases}
1, & \text{if } T_{\text{enter}, i} \leq t < T_{\text{enter}, i} + D_i \\
0, & \text{otherwise}
\end{cases}
```
where:
- (A_{i,t}) is a binary mask indicating whether job (i) is active at time (t).

5. Final Resource Utilization- The 3D array representing job resource utilization is calculated as:
```math
J_{i,k,t} = A_{i,t} \cdot R_{i,k}
```
where:
```math
(J_{i,k,t}) is the resource demand of job (i) for resource (k) at time (t).
```
```math
(A_{i,t}) ensures resource demands are applied only during active job periods.
```
## Features

- **Customizable Cluster Configuration**:
  - Number of machines
  - Number of jobs
  - Number of resources (e.g., CPU, RAM, Disk, Network)
  - Maximum number of ticks (time steps)

- **Observation Space**:
  - Machine free space
  - Job utilization
  - Job status

- **Action Space**:
  - Time tick action (whether to increment time)
  - Machine index for scheduling
  - Job index for scheduling

- **Metrics**:
  - Tracks job states: Not Created, Pending, Running, Completed
  - Ensures proper resource management across machines

---

## Installation

1. Install Python dependencies:
   ```bash
   pip install gymnasium numpy pydantic typing-extensions