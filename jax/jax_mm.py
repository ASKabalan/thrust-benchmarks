import jax
import jax.numpy as jnp
import time
import argparse
import os


@jax.jit
def gemm(a,b ,c):
    d = jnp.matmul(a,b) + c

    return jnp.matmul(c,d) + d
    
def main(m, n, k, precision, output):

    a = jax.random.normal(jax.random.PRNGKey(0), shape=(m, k))
    b = jax.random.normal(jax.random.PRNGKey(1), shape=(k, n))
    c = jax.random.normal(jax.random.PRNGKey(2), shape=(m, n))

    print(a.shape)

    gemm(a,b,c).block_until_ready()

    start = time.perf_counter()
    gemm(a,b,c).block_until_ready()
    end = time.perf_counter()

    elapsed = (end - start) * 1000

    def compute_effective_bandwidth(m, n, k, latency):
        return ((m * k + k * n + m * n) * 4) / (latency * 1e-3) / 1e9

    def compute_effective_tflops(m, n, k, latency):
        return (2.0 * m * k * n) / (latency * 1e-3) / 1e12


    bandwidth = compute_effective_bandwidth(4096, 4096, 4096, elapsed)
    tflops = compute_effective_tflops(4096, 4096, 4096, elapsed)

    print(f"Elapsed time: {elapsed:.2f} ms")
    print(f"Effective bandwidth: {bandwidth:.2f} GB/s")
    print(f"Effective TFLOPS: {tflops:.2f} TFLOPS")

    if output is not None:
        if not os.path.exists(output):
            with open(output, "w") as f:
                f.write("framework,bandwidth,tflops,precision,m,n,k,time\n")
        with open(output, "w") as f:
            f.write(f"JAX,{bandwidth:.2f},{tflops:.2f},{precision},{m},{n},{k},{elapsed:.2f}\n")

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m" , "--m", type=int, default=4096)
    argparser.add_argument("-n" , "--n", type=int, default=4096)
    argparser.add_argument("-k" , "--k", type=int, default=4096)
    argparser.add_argument("-p" , "--precision", type=str, default="fp32")
    argparser.add_argument("-xla" , "--optimize_xla", action="store_true")
    argparser.add_argument("-o" , "--output", type=str, default=None)

    args = argparser.parse_args()

    if argparse.optimize_xla:
        os.environ['XLA_FLAGS'] = (
                    '--xla_gpu_enable_triton_softmax_fusion=true '
                    '--xla_gpu_triton_gemm_any=True '
                    '--xla_gpu_enable_async_collectives=true '
                    '--xla_gpu_enable_latency_hiding_scheduler=true '
                    '--xla_gpu_enable_highest_priority_async_stream=true '
                )


    if args.precision == "fp32":
        jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_enable_x64", True)
    
    main(args.m, args.n, args.k, args.precision, args.output)



