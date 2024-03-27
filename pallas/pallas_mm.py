
from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from jax import lax
import time
import argparse
import os

def matmul_kernel(x_ref, y_ref, z_ref, *, bk: int):
  m, n = x_ref.shape[0], y_ref.shape[1]
  k = y_ref.shape[0]
  acc = jnp.zeros((m, n), dtype=jnp.float32)
  def body(i, acc):
    x_k = pl.load(x_ref, (slice(None), pl.ds(i * bk, bk)))
    y_k = pl.load(y_ref, (pl.ds(i * bk, bk), slice(None)))
    return acc + x_k @ y_k
  z_ref[...] = lax.fori_loop(0, k // bk, body, acc).astype(z_ref.dtype)


@partial(jax.jit, static_argnames=('debug', 'interpret','bn', 'bk', 'bm'))
def matmul(x: jax.Array, y: jax.Array, *, bm: int = 32, bn: int = 32, bk: int = 32,
           debug: bool = False, interpret: bool = False):
  m, n, k = x.shape[0], y.shape[1], y.shape[0]
  grid = (m // bm, n // bn)
  return pl.pallas_call(
    partial(matmul_kernel, bk=bk),
    out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
    grid=grid,
    in_specs=[
      pl.BlockSpec(lambda i, j: (i, 0), (bm, k)),
      pl.BlockSpec(lambda i, j: (0, j), (k, bn))
    ],
    out_specs=pl.BlockSpec(lambda i, j: (i, j), (bm, bn)),
    debug=debug, interpret=interpret,
  )(x, y)

def autotune(x , y , c, precision, dtype, output):

  times = {}
  for bm in [16 , 32 , 64, 128, 256]:
      for bn in [16 , 32 , 64, 128, 256]:
          for bk in [16 , 32 , 64, 128, 256]:

              print(f"Trying bm={bm}, bn={bn}, bk={bk}")
              # 260000 is my shared memory limit
              # TODO(wassim) : make it more dynamic
              if bm * bn * bk > 260000:
                  continue

              z = matmul(x, y, bm=bm, bn=bn, bk=bk).block_until_ready()
              start = time.perf_counter()
              z = matmul(x, y, bm=bm, bn=bn, bk=bk).block_until_ready()
              end = time.perf_counter()
              elapsed = (end - start)*1000
              times[elapsed] = (bm, bn, bk)

  best_time = min(times.keys())
  bm, bn, bk = times[best_time]
  return bm, bn, bk

def main(m, n, k, precision, dtype, output):

  k1, k2 = jax.random.split(jax.random.PRNGKey(0))
  x = jax.random.normal(k1, (m, k),dtype=dtype)
  y = jax.random.normal(k2, (k, n),dtype=dtype)
  c = jax.random.normal(k2, (m, n),dtype=dtype)

  bm, bn, bk = autotune(x, y,c, precision, dtype, output)

  z = matmul(x, y, bm=bm, bn=bn, bk=bk).block_until_ready()
  start = time.perf_counter()
  z = matmul(x, y, bm=bm, bn=bn, bk=bk).block_until_ready()
  end = time.perf_counter()
  elapsed = (end - start)*1000

  def compute_effective_bandwidth(m, n, k, latency):
    return ((m * k + k * n + m * n) * 4) / (latency * 1e-3) / 1e9

  def compute_effective_tflops(m, n, k, latency):
      return (2.0 * m * k * n) / (latency * 1e-3) / 1e12

  bandwidth = compute_effective_bandwidth(m, n, k, elapsed)
  tflops = compute_effective_tflops(m, n, k, elapsed)
  
  if output is not None:
    with open(output, "w") as f:
      if not os.path.exists(output):
        f.write("framework,bandwidth,tflops,precision,m,n,k,time\n")
      f.write(f"PALLAS,{bandwidth:.2f},{tflops:.2f},{precision},{m},{n},{k},{elapsed:.2f}\n")


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m" , "--m", type=int, default=4096)
    argparser.add_argument("-n" , "--n", type=int, default=4096)
    argparser.add_argument("-k" , "--k", type=int, default=4096)
    argparser.add_argument("-p" , "--precision", type=str, default="fp32")
    argparser.add_argument("-o" , "--output", type=str, default=None)

    args = argparser.parse_args()

    if args.precision == "fp32":
        dtype = jnp.float32
    elif args.precision == "fp64":
        dtype = jnp.float64
    
    main(args.m, args.n, args.k, args.precision, dtype, args.output)

