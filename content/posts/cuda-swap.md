---
title: "Building a GPU Memory Swapper with LD_PRELOAD"
date: 2026-05-25
tags: ["cuda", "gpu", "memory", "systems"]
summary: "Notes on building cuda_swap: an LD_PRELOAD library that transparently spills GPU memory to host RAM using CUDA unified memory."
---

I wanted to see if GPU memory overflow could be handled transparently: no code changes, no application flags, just drop in a library and have it work. This is a writeup of how that went, including two dead ends before finding the right approach.

The result is [cuda_swap](https://github.com/ali-alshaar7/cuda_swap).

---

## The Interception Layer

`LD_PRELOAD` loads a shared library before anything else in the process. If that library exports a symbol that another library also exports, the preloaded one wins. CUDA programs allocate GPU memory through two APIs: `cudaMalloc` in the runtime layer and `cuMemAlloc` in the driver layer. Most frameworks use the driver API directly, or the runtime API which calls down to it. Exporting both from a preloaded `.so` means every allocation the process makes goes through my code first.

The interception itself is straightforward. The trickier part is resolving the real underlying functions: I need a pointer to the real allocator so the hook can delegate to it.

`dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")` searches past this library in the load order to get glibc's own `dlsym`, pinned by version string. Then `libcuda.so.1` is opened with `RTLD_NOLOAD` to get a handle to the already-loaded library without loading a second copy. Resolving from that specific handle bypasses `RTLD_NEXT` search order entirely, so the result is always the symbol from `libcuda.so.1` regardless of what else is in the preload chain:

```cpp
using DlsymFn = void*(*)(void*, const char*);
DlsymFn real_dlsym = (DlsymFn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");

// RTLD_NOLOAD gets a handle to the already-loaded libcuda without a second dlopen
void* libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
real_cuMemAlloc = (Fn_cuMemAlloc)real_dlsym(libcuda, "cuMemAlloc_v2");
```

With that in place, I control every allocation and can decide what to do with it.

---

## Dead End 1: Manual Eviction

The first approach was to track all live allocations and evict them manually when VRAM fills up. When a new allocation request comes in and there isn't enough VRAM:

1. Pick a buffer to evict (LRU or similar)
2. Copy it to host RAM with `cudaMemcpy`
3. Free the VRAM
4. Service the new allocation
5. When the evicted buffer is needed again, copy it back

This is exactly what the Linux kernel does with anonymous pages and swap. The kernel knows when to bring a page back because the CPU's MMU raises a hardware page fault when a process touches an address whose physical page has been swapped out. The fault is invisible to the application; the kernel handles it, copies the page back, and execution continues.

The GPU doesn't expose this. There's no public mechanism for a userspace library to receive a notification when the GPU is about to dereference a specific device pointer. Without that, there's no way to know when to restore an evicted buffer before the GPU needs it. The only alternative is to intercept every CUDA kernel launch and inspect its arguments, but kernel arguments are passed to the driver as an opaque array with no public API to decode which entries are device pointers vs scalars. The approach doesn't have a clean path forward.

---

## Dead End 2: Virtual Memory Management

The CUDA Virtual Memory Management API (CUDA 10.2+) separates virtual address reservation from physical backing, letting you remap what physical memory a pointer points to without changing the pointer itself. This looked promising: if you could remap an evicted buffer's backing without changing its pointer, the pointer-invalidation problem from manual eviction goes away.

But the fundamental constraint is the same: you still need to know *before* a kernel runs which buffers it will touch, so you can ensure they're in VRAM when the kernel starts. VMM gives finer-grained control over the mechanics of that operation without solving the scheduling problem.

The root issue with any manual approach is the missing primitive: hardware notification when the GPU accesses an address that isn't backed by VRAM.

---

## The Missing Piece: cuMemAllocManaged

After both dead ends I came across [this thread on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1di3kyn/long_shot_solution_combine_system_ram_with_vram/) discussing VRAM extension for llama.cpp and Stable Diffusion. The solutions there required modifying application code or setting software-specific flags. But the thread mentioned CUDA unified memory, which led to `cuMemAllocManaged`, and that has exactly the hardware support I was missing.

### How It Works

`cuMemAllocManaged` allocates memory in a shared virtual address space accessible from both CPU and GPU. The driver handles migration automatically using the GPU's hardware page fault mechanism, available on Pascal-generation GPUs (GTX 10 series) and later.

When you allocate managed memory, the driver maps the virtual address range into both the CPU's and GPU's page tables, but initially no physical memory is allocated at all: pages have no backing until they're first touched. The first time the GPU touches a page, the hardware MMU raises a fault. The CUDA driver's fault handler runs on the CPU, allocates a physical VRAM page, DMAs any initialised content in, updates the GPU page table to mark it present, and signals the GPU to replay the faulting instruction. From the kernel's perspective, the instruction stalls until the fault is serviced, then completes normally.

Pages move rather than copy. When a page migrates to VRAM, the host physical page is freed; when it's later evicted from VRAM under pressure, a fresh host physical page is allocated for it. Total physical backing at any moment is at most 1× the allocation size, split across wherever the pages currently are.

Before Pascal, unified memory worked differently: all pages were migrated eagerly to VRAM before any kernel launched, and the total managed allocation had to fit in VRAM or the launch failed. Hardware page faults changed this to genuine demand paging, which is what makes the overflow case work at all.

`cuMemAdvise` influences the driver's migration policy:

```cpp
// Prefer keeping this allocation in VRAM; only evict under genuine pressure
cuMemAdvise(ptr, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device);
// Map this allocation on the device without migrating (for read-mostly access patterns)
cuMemAdvise(ptr, size, CU_MEM_ADVISE_SET_ACCESSED_BY, device);
```

`PREFERRED_LOCATION` doesn't pin pages: the driver can still evict under pressure, but it biases the policy to keep pages on the GPU when there's room.

### The Memory Cost

In principle, managed memory pages on Pascal+ GPUs are demand-paged: physical backing is allocated on first touch, and pages that migrate to VRAM free their host physical backing. In practice, `cuMemAllocManaged` with `CU_MEM_ATTACH_GLOBAL` (the flag required for GPU-wide accessibility) places pages in system RAM at allocation time. And because GPU kernels execute asynchronously, a Python allocation loop can issue many managed allocations before the GPU has executed any of them, leaving all N tensors simultaneously backed in system RAM.

This matters. Routing every allocation through managed memory when VRAM has room can exhaust host RAM from this transient in-flight window alone, before any real overflow has occurred. The threshold approach sidesteps this: `cuMemAlloc` pages are placed directly in VRAM, never touch host RAM, and cannot be evicted by the managed memory driver.

For managed allocations that do get used, the code also calls `cuMemPrefetchAsync` immediately after the `cuMemAdvise` calls, while VRAM still has headroom, to proactively migrate pages before GPU kernels queue up and start faulting:

```cpp
if (free_vram() > size)
    cuMemPrefetchAsync(*out, size, device, 0);
```

This reduces runtime page faults when the workload has predictable access patterns.

---

## The Threshold Allocator

The approach that worked: use regular `cuMemAlloc` while VRAM has headroom, and only fall back to `cuMemAllocManaged` when it doesn't. Most of the time, most allocations are regular VRAM: fast, no driver migration machinery involved, no host RAM backing store consumed. Only allocations that actually overflow use managed memory.

```cpp
bool use_managed = free_vram() < threshold + requested_size;

if (!use_managed) {
    rc = real_cuMemAlloc(out, size);
    if (rc != CUDA_SUCCESS) use_managed = true;  // VRAM full despite check, try managed
} else if (free_vram() >= size) {
    // Threshold says managed, but VRAM can still fit this allocation.
    // Prefer cuMemAlloc to avoid system RAM pressure from managed page placement.
    rc = real_cuMemAlloc(out, size);
    if (rc == CUDA_SUCCESS) use_managed = false;
}
if (use_managed) {
    rc = real_cuMemAllocManaged(out, size, CU_MEM_ATTACH_GLOBAL);
    real_cuMemAdvise(*out, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device);
    real_cuMemAdvise(*out, size, CU_MEM_ADVISE_SET_ACCESSED_BY, device);
    if (free_vram() > size)
        real_cuMemPrefetchAsync(*out, size, device, 0);
}
```

The threshold (default 512 MB) prevents switching to managed right at the edge of VRAM: the last 512 MB stays as a buffer. This matters because the managed memory infrastructure itself has overhead, and because two concurrent allocations that both observe "just enough room" could both fail without the buffer.

Running the OOM test on an RTX 3070 Ti (8 GB VRAM, 7.65 GB reported free by CUDA after driver reserves) in a 9 GB Docker container, `THRESHOLD=512` and `THRESHOLD=999999` produce nearly identical peak system RAM: both are bounded by the true overflow amount (workload size minus VRAM capacity). The `else if` guard above is what makes this true: even with an absurdly high threshold, allocations that fit in VRAM still go through `cuMemAlloc` and never touch host RAM.

The difference shows up in throughput. Across repeated trials, threshold-based allocation runs about 4% faster per tensor. `cuMemAlloc` pages are pinned in VRAM and cannot be evicted to service page faults from managed allocations: VRAM-resident tensors stay put during computation. `cuMemAllocManaged` pages, even when currently resident in VRAM, are always fair game for eviction, which adds migration round-trips during compute-heavy operations that touch the full working set.

I also check `/proc/meminfo` on each allocation that crosses the threshold, to make sure host RAM can actually absorb the overflow. On systems with cgroup memory limits (Docker containers), I read the limit directly from `/sys/fs/cgroup/memory.max` since `/proc/meminfo` reflects host totals, not the container limit. A 2 GB safety margin guards against exhausting host RAM under pressure.

---

## JAX and the BFC Allocator

After PyTorch worked, I tried JAX. It failed with an XLA memory error during initialisation, before any user code ran. Understanding why requires looking at how the two frameworks actually talk to the CUDA allocator.

[PyTorch's caching allocator](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) calls `cudaMalloc` and `cudaFree` frequently, but caches freed blocks rather than immediately returning them to the driver. Each `torch.Tensor` allocation goes through the allocator, which either carves a block out of its cache or calls `cudaMalloc` if nothing fits. The key point: individual tensor allocations reach `cudaMalloc`, so my hook sees them. PyTorch also runs eagerly: Python code executes top to bottom, tensors are allocated and freed as the interpreter encounters them, and the live set at any moment is just what that point in the code has in scope.

JAX's XLA backend works differently in both dimensions. At startup, XLA calls `cudaMalloc` once for a large slab (typically 90% of reported VRAM) and manages sub-allocations within that slab using a BFC (Best-Fit with Coalescing) algorithm. Individual array allocations never reach `cudaMalloc`. When XLA runs out of space in the slab, it reports OOM internally before ever calling `cudaMalloc` for more. My hook only sees the one initial slab allocation, which succeeds before VRAM is under any pressure. From cuda_swap's perspective the process made one allocation, it succeeded, and then JAX internally OOMs.

The fix:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

This tells XLA to bypass the BFC slab and call `cudaMalloc` per tensor. Every allocation now goes through the hook. The tradeoff is that the BFC allocator is more efficient for the common case (amortised allocation cost, better fragmentation handling), but platform mode is required for interception.

Even with this flag, JAX uses more memory than PyTorch for equivalent operations. XLA traces and compiles the full computation graph before executing any of it. During compilation, XLA may keep multiple representations of intermediate values alive simultaneously. PyTorch's eager model doesn't have this: each operation executes immediately, and the only tensors alive are the ones Python currently holds references to.

---

## The Training Case

Synthetic allocation tests (allocate N tensors, keep them alive, verify) are a clean way to confirm the basic mechanism works. Training has an additional pattern worth testing: the activation spike.

During a forward pass, input activations at each layer are retained for the backward pass; gradient computation requires them. With large hidden dimensions, multiple activation tensors can be simultaneously alive at peak even if the model weights themselves fit in VRAM. The spike is temporary (activations are freed once backward completes), but its peak can significantly exceed the steady-state VRAM footprint.

The test uses a `8192→8192→1` linear network with batch size tuned so that the three hidden-layer activations together exceed VRAM:

```
model weights:            ~0.5 GB
input tensor:             ~2 GB
hidden layer 1 activations (kept for backward): ~2 GB
hidden layer 2 activations (kept for backward): ~2 GB
──────────────────────────────────────────────────────
peak during forward/backward: ~6.5 GB  (on an 8 GB GPU)
```

Without cuda_swap this OOMs on the first forward pass. With it, the overflow activations spill to host RAM and are faulted back in as backward proceeds. The cost is proportional to how much had to spill: each activation tensor that migrated has to be DMA'd back during backward.

---

## Further Reading

- [CUDA Unified Memory Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming): full spec for `cuMemAllocManaged`, `cuMemAdvise`, and the pre-Pascal vs Pascal migration model
- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management): how the caching allocator works, when it calls `cudaMalloc`, and fragmentation behaviour
- [BFC Allocator source (TensorFlow)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/bfc_allocator.h): the slab allocator XLA inherited; XLA originated as part of TensorFlow before being extracted into its own project
- [r/LocalLLaMA thread on VRAM + RAM](https://www.reddit.com/r/LocalLLaMA/comments/1di3kyn/long_shot_solution_combine_system_ram_with_vram/): the discussion that pointed toward unified memory
