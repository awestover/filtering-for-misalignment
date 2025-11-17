# Parallelization Concepts Explained

## Three Ways to Run Code in Parallel

### 1. **Async I/O** (What your script already uses)
```
Single Process, Single Thread, Multiple Tasks

Process 1
  ├─ Task A: Fetch URL 1 ──── (waiting for network)
  ├─ Task B: Fetch URL 2 ──── (waiting for network)
  ├─ Task C: Fetch URL 3 ──── (waiting for network)
  └─ Task D: Fetch URL 4 ──── (waiting for network)
       ↓
  While waiting, switch between tasks efficiently
```

**How it works:**
- Single Python process
- When Task A is waiting for network, Python switches to Task B
- Like a chef cooking multiple dishes: while one is baking, prep another
- **Best for I/O-bound work** (network, disk)

**In your script:**
```python
async def _fetch_all_texts(urls, concurrency=500):
    # Downloads 500 URLs concurrently in ONE process
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(fetch(url) for url in urls))
```

**Limitations:**
- Still only uses ONE CPU core
- Can't parallelize the *search queries* themselves (only the downloads within each query)
- Python's GIL (Global Interpreter Lock) limits true parallelism


### 2. **Multiprocessing** (What I just added)
```
Multiple Processes, Each Running Independently

Main Process
  ├─ Worker Process 1: Handles queries 1-5000
  ├─ Worker Process 2: Handles queries 5001-10000
  ├─ Worker Process 3: Handles queries 10001-15000
  └─ Worker Process 4: Handles queries 15001-18564
       ↓
  Each worker runs on its own CPU core
  Each worker has its own memory space
  Workers cannot share variables directly
```

**How it works:**
- Creates separate Python processes (like running the script 4 times)
- Each process has its own memory and Python interpreter
- OS schedules processes on different CPU cores
- **Best for CPU-bound work** OR parallelizing independent I/O tasks

**In your updated script:**
```python
with multiprocessing.Pool(processes=4) as pool:
    results = pool.starmap(_worker_process_queries, [
        (terms[0:5000], 0, ...),    # Worker 1
        (terms[5000:10000], 1, ...), # Worker 2
        (terms[10000:15000], 2, ...), # Worker 3
        (terms[15000:18564], 3, ...), # Worker 4
    ])
```

**Key differences from async:**
- Uses multiple CPU cores (async uses one)
- Each worker is completely independent
- Processes communicate by passing data (not sharing memory)
- More overhead than async (creating processes is expensive)


### 3. **Subprocess** (Different from multiprocessing!)
```
Running External Programs

Your Python Script
  ├─ subprocess.run(["curl", "https://example.com"])
  ├─ subprocess.run(["git", "commit", "-m", "message"])
  └─ subprocess.run(["ffmpeg", "-i", "input.mp4", "output.mp4"])
       ↓
  Launches external programs, waits for them to finish
```

**How it works:**
- Runs **other programs** (not Python code)
- Like opening Terminal and typing a command
- Your Python script waits for the command to finish

**Examples:**
```python
import subprocess

# Run git command
subprocess.run(["git", "status"])

# Run shell command
subprocess.run(["ls", "-la"])

# Run with output capture
result = subprocess.run(["curl", "https://example.com"],
                       capture_output=True)
```

**When to use:**
- Need to run non-Python programs
- Need shell commands
- **NOT for parallelizing Python code**


## Comparison Table

| Feature | Async I/O | Multiprocessing | Subprocess |
|---------|-----------|-----------------|------------|
| CPU cores used | 1 | Multiple | Varies |
| Memory overhead | Low | High | High |
| Best for | I/O waiting | CPU work or independent I/O | Running external programs |
| Can share variables? | Yes | No (must pass data) | No |
| Startup cost | Very low | Medium-high | High |
| Python GIL bypass? | No | Yes | N/A |
| Example use case | Download 500 URLs | Process 4 search queries at once | Run git/curl/ffmpeg |


## Your Script Uses BOTH!

### Current Implementation (Hybrid Approach):

```
Main Process (parent)
  │
  ├─ Worker Process 1 ───────────────────────────────┐
  │    │                                              │
  │    └─ Query "AI safety" ────────────────────────┐│
  │         ├─ Search API (1 request)                ││
  │         └─ Async download 500 URLs concurrently  ││  ← Async I/O
  │                                                   ││
  ├─ Worker Process 2 ───────────────────────────────┘│
  │    │                                               │  ← Multiprocessing
  │    └─ Query "ML alignment" ─────────────────────┐ │
  │         ├─ Search API (1 request)                │ │
  │         └─ Async download 500 URLs concurrently  │ │
  │                                                   │ │
  └─ Worker Process 3 ... (and so on)                │ │
       └─ Query "reward hacking" ────────────────────┘ │
            ├─ Search API (1 request)                  │
            └─ Async download 500 URLs concurrently ───┘
```

**Benefits:**
1. **Multiprocessing** parallelizes search queries across CPU cores
2. **Async I/O** parallelizes URL downloads within each query
3. Best of both worlds!


## How to Use Your Updated Script

### Sequential (old way - still works):
```bash
python3 1searchterm-download.py
# Processes queries one at a time
```

### Parallel with 4 workers (NEW!):
```bash
python3 1searchterm-download.py --workers 4
# Processes 4 queries simultaneously!
```

### Parallel with 8 workers:
```bash
python3 1searchterm-download.py --workers 8
# Processes 8 queries simultaneously!
# Recommended for 12-16 core machines
```


## Performance Estimates

With your 18,564 search terms:

| Workers | Est. Time | Speedup | CPU Usage |
|---------|-----------|---------|-----------|
| 1 (sequential) | 2-4 days | 1x | ~20% (1 core busy) |
| 4 workers | 12-24 hours | ~3-4x | ~80% (4 cores busy) |
| 8 workers | 6-12 hours | ~6-7x | ~160% (8 cores busy) |
| 16 workers | 4-8 hours | ~8-10x | Diminishing returns from API limits |

**Note:** You won't get perfect linear scaling (8 workers ≠ 8x faster) because:
- API rate limits still apply
- Network bandwidth is shared
- Some queries fail and need retries


## Why Not Just Use Subprocess?

You **could** manually do:
```bash
python3 1searchterm-download.py --terms-file chunk1.txt &
python3 1searchterm-download.py --terms-file chunk2.txt &
python3 1searchterm-download.py --terms-file chunk3.txt &
python3 1searchterm-download.py --terms-file chunk4.txt &
```

**But multiprocessing is better because:**
- ✓ Cleaner: just `--workers 4` instead of manual script splitting
- ✓ Automatic work distribution
- ✓ Results are aggregated for you
- ✓ Better error handling
- ✓ No need to manually split search_terms.md
- ✓ Progress tracking works correctly


## Summary

**Async I/O:** "Do 500 things at once while waiting for network" (1 core)
**Multiprocessing:** "Run 4 copies of my code on 4 CPU cores" (4+ cores)
**Subprocess:** "Run git/curl/other programs from Python" (not for parallelizing Python)

Your script now uses **multiprocessing** to run multiple search queries in parallel,
and each worker uses **async I/O** to download multiple URLs concurrently!
