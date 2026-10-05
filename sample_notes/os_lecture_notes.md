# CS301 Operating Systems — Lecture Notes

## Unit 1: Processes

A process is a program in execution. A program is a passive entity stored on disk, while a process is an active entity with a program counter, registers, a stack and a heap.

### Process States

A process moves through five states during its lifetime:

- New: the process is being created.
- Ready: the process is waiting to be assigned to the CPU.
- Running: instructions are being executed.
- Waiting: the process is waiting for an event such as I/O completion.
- Terminated: the process has finished execution.

### Process Control Block

The operating system represents each process with a Process Control Block (PCB). The PCB stores the process state, program counter, CPU registers, scheduling information, memory-management information and I/O status. During a context switch the kernel saves the state of the old process in its PCB and loads the saved state of the new process. Context-switch time is pure overhead because the system does no useful work while switching.

## Unit 2: CPU Scheduling

The short-term scheduler selects a process from the ready queue and allocates the CPU to it. Scheduling criteria covered in this course are CPU utilisation, throughput, turnaround time, waiting time and response time.

### First-Come, First-Served (FCFS)

FCFS allocates the CPU in the order processes arrive. It is non-preemptive and simple to implement with a FIFO queue. FCFS suffers from the convoy effect: short processes wait behind one long CPU-bound process, increasing average waiting time.

### Shortest Job First (SJF)

SJF picks the process with the smallest next CPU burst. SJF is provably optimal for minimum average waiting time, but the length of the next CPU burst is not known in advance and must be predicted using exponential averaging: tau(n+1) = alpha * t(n) + (1 - alpha) * tau(n). The preemptive version of SJF is called Shortest Remaining Time First (SRTF).

### Round Robin (RR)

Round Robin gives each process a fixed time quantum, typically 10 to 100 milliseconds. When the quantum expires the process is preempted and added to the tail of the ready queue. If the quantum is very large, RR behaves like FCFS; if it is very small, context-switch overhead dominates. As a rule of thumb, 80 percent of CPU bursts should be shorter than the time quantum.

### Priority Scheduling

Each process is assigned a priority and the CPU goes to the highest-priority process. The main problem is starvation, where low-priority processes may never execute. The solution is aging: gradually increasing the priority of processes that wait in the system for a long time.

## Unit 3: Deadlocks

A deadlock is a situation in which a set of processes are blocked because each process is holding a resource and waiting for a resource held by another process in the set.

### Necessary Conditions

A deadlock can arise only if four conditions hold simultaneously (the Coffman conditions):

1. Mutual exclusion: at least one resource is held in a non-sharable mode.
2. Hold and wait: a process holds at least one resource while waiting for others.
3. No preemption: resources cannot be forcibly taken from a process.
4. Circular wait: a closed chain of processes exists where each waits for a resource held by the next.

### Banker's Algorithm

The Banker's algorithm is a deadlock-avoidance algorithm. Each process declares the maximum number of instances of each resource type it may need. The system grants a request only if the resulting state is safe, meaning there exists a safe sequence in which every process can obtain its maximum need and finish. The data structures used are Available, Max, Allocation and Need, where Need = Max - Allocation.

## Unit 4: Memory Management

### Paging

Paging divides physical memory into fixed-size blocks called frames and logical memory into blocks of the same size called pages. A page table maps each page number to a frame number. Paging eliminates external fragmentation but can cause internal fragmentation in the last page of a process. A logical address is split into a page number p and a page offset d.

### Translation Lookaside Buffer

A TLB is a small, fast associative cache of recent page-table entries. On a TLB hit the frame number is available immediately; on a TLB miss the page table in memory must be consulted. Effective access time = hit ratio * (TLB time + memory time) + (1 - hit ratio) * (TLB time + 2 * memory time).

### Segmentation

Segmentation divides a program into variable-sized logical units such as code, data and stack. Each logical address is a pair (segment number, offset), and a segment table stores the base and limit of each segment. Segmentation matches the programmer's view of memory but suffers from external fragmentation.

### Page Replacement

When a page fault occurs and no frame is free, a victim page must be replaced. FIFO replaces the oldest page and can suffer from Belady's anomaly, where adding frames increases the number of page faults. The Optimal algorithm replaces the page that will not be used for the longest time; it gives the lowest fault rate but cannot be implemented in practice. LRU replaces the page that has not been used for the longest time and does not suffer from Belady's anomaly.
