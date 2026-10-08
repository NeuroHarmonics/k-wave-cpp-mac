// Time VkFFT out-of-place R2C and C2R as used by kspaceFirstOrder-Metal, with tuning options from the environment.
// Prints the GPU time per transform and the number of passes over memory (uploads) per axis.
//
// Usage: ./fftbench nx ny [nz]
// Environment: COALESCED (coalescedMemory in bytes), AIM (aimThreads), REGBOOST (registerBoost), LUT (useLUT),
//              NOMERGE (disableMergeSequencesR2C), BWBOOST (performBandwidthBoost), GROUP0..2 (groupedBatch),
//              SHMEM (sharedMemorySize), ITERS (transforms timed, default 50)
// Example: COALESCED=16 ./fftbench 2048 2048
#define VKFFT_BACKEND 5
#include <vkFFT.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
static long env(const char* n, long d) { const char* v = getenv(n); return v ? atol(v) : d; }
int main(int argc, char** argv)
{
  uint64_t nx = atol(argv[1]), ny = atol(argv[2]), nz = argc > 3 ? atol(argv[3]) : 1, nxR = nx / 2 + 1;
  MTL::Device* dev = MTL::CreateSystemDefaultDevice();
  MTL::CommandQueue* q = dev->newCommandQueue();
  uint64_t rb = nx * ny * nz * 4, cb = nxR * ny * nz * 8;
  MTL::Buffer* real = dev->newBuffer(rb, MTL::ResourceStorageModeShared);
  MTL::Buffer* cplx = dev->newBuffer(cb, MTL::ResourceStorageModeShared);
  float* r = (float*)real->contents(); for (uint64_t i = 0; i < nx * ny * nz; i++) r[i] = float(rand()) / RAND_MAX;
  VkFFTApplication app[2] = {};
  for (int inv = 0; inv < 2; inv++)
  {
    VkFFTConfiguration c = {};
    c.FFTdim = nz > 1 ? 3 : 2; c.size[0] = nx; c.size[1] = ny; c.size[2] = nz;
    c.performR2C = 1; c.isInputFormatted = 1; c.inverseReturnToInputBuffer = 1;
    c.inputBufferStride[0] = nx; c.inputBufferStride[1] = nx * ny; c.inputBufferStride[2] = nx * ny * nz;
    c.bufferStride[0] = nxR; c.bufferStride[1] = nxR * ny; c.bufferStride[2] = nxR * ny * nz;
    c.makeForwardPlanOnly = !inv; c.makeInversePlanOnly = inv;
    c.device = dev; c.queue = q; c.buffer = &cplx; c.bufferSize = &cb; c.inputBuffer = &real; c.inputBufferSize = &rb;
    if (env("AIM", 0)) c.aimThreads = env("AIM", 0);
    if (env("COALESCED", 0)) c.coalescedMemory = env("COALESCED", 0);
    if (env("REGBOOST", 0)) { c.registerBoost = env("REGBOOST", 0); c.registerBoostNonPow2 = 1; }
    if (env("LUT", 0)) c.useLUT = env("LUT", 0);
    if (env("NOMERGE", 0)) c.disableMergeSequencesR2C = 1;
    if (env("BWBOOST", 0)) c.performBandwidthBoost = env("BWBOOST", 0);
    for (int d = 0; d < 3; d++) { char n[8]; snprintf(n, 8, "GROUP%d", d); if (env(n, 0)) c.groupedBatch[d] = env(n, 0); }
    if (env("SHMEM", 0)) c.sharedMemorySize = env("SHMEM", 0);
    VkFFTResult res = initializeVkFFT(&app[inv], c);
    if (res) { printf("init failed %d\n", res); return 1; }
  }
  const int iters = env("ITERS", 50);
  double t[2];
  for (int inv = 0; inv < 2; inv++)
    for (int rep = 0; rep < 2; rep++)   // first repetition is a warm up
    {
      MTL::CommandBuffer* buf = q->commandBuffer();
      MTL::ComputeCommandEncoder* enc = buf->computeCommandEncoder();
      VkFFTLaunchParams lp = {}; lp.commandBuffer = buf; lp.commandEncoder = enc; lp.buffer = &cplx; lp.inputBuffer = &real;
      for (int i = 0; i < iters; i++) VkFFTAppend(&app[inv], inv ? 1 : -1, &lp);
      enc->endEncoding(); buf->commit(); buf->waitUntilCompleted();
      t[inv] = (buf->GPUEndTime() - buf->GPUStartTime()) / iters;
    }
  printf("%llux%llux%llu  R2C %.3f ms  C2R %.3f ms  (axes uploads", nx, ny, nz, 1e3 * t[0], 1e3 * t[1]);
  for (int d = 0; d < 3; d++) printf(" %llu", (unsigned long long)app[0].localFFTPlan->numAxisUploads[d]);
  printf(")\n");
}
