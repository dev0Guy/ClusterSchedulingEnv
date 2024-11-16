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
#### Job:

#### Machines:

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