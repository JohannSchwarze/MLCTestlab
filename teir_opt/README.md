# README


## Cache Sizes

- L1 Data (PC): 128 KiB
- L2 Data (PC): 16 MiB

- L1 Data (EC): 64 KiB
- L2 Data (EC): 4 MiB

```
sysctl hw.perflevel0.l1icachesize
sysctl hw.perflevel0.l1dcachesize
sysctl hw.perflevel0.l2cachesize

sysctl hw.perflevel1.l1icachesize
sysctl hw.perflevel1.l1dcachesize
sysctl hw.perflevel1.l2cachesize
```

- `perflevel0`: Performance Core
- `perflevel1`: Efficiency Core








## Primitive Einsums

```

C = A * B
km,nk->nm

C = A * B^T
km,kn->nm

```








## Einsums

```

v1:
(a=8, b=64, c=24, d=4, e=12, f=256)
abcd,ebf->aefcd

v2:
(a=8, b=64, e=12, f=256, g=96)
abg,ebf->aefg

v3:
(a=8, b=64, e=12, g=96, h=4, i=64)
abg,ebhi->aehig

v4:
(a=8, b=64, e=12, h=4, i=64, j=3, k=32)
abjk,ebhi->aehijk

```






## TEIR

```
teir @v1 {
    tensor %in0: f32
    tensor %in1: f32
    tensor %out: f32

    axis @a extent 8   strides { in0: 24576, in1: 0,     out: 1179648 }
    axis @b extent 64  strides { in0: 384,   in1: 1024,  out: 0 }
    axis @c extent 24  strides { in0: 16,    in1: 0,     out: 16 }
    axis @d extent 4   strides { in0: 4,     in1: 0,     out: 4 }
    axis @e extent 12  strides { in0: 0,     in1: 65536, out: 98304 }
    axis @f extent 256 strides { in0: 0,     in1: 4,     out: 384 }

    primitive @zero : Zero axes { M: [], N: [] } metadata { data_type: f32 }
    primitive @contract : Contraction axes { M: [], N: [], K: [] } metadata { data_type: f32}

    schedule {
        roots: [@iter_a]

        iter @iter_a axis @a policy sequential children [@iter_b]
        iter @iter_b axis @b policy sequential children [@iter_c]
        iter @iter_c axis @c policy sequential children [@iter_d]
        iter @iter_d axis @d policy sequential children [@iter_e]
        iter @iter_e axis @e policy sequential children [@iter_f]
        iter @iter_f axis @f policy sequential children [@inv_zero, @inv_gemm]

        invoke @inv_zero primitive @zero guard first(@b)
        invoke @inv_gemm primitive @gemm
    }
}
```

```
teir @v2 {
    tensor %in0: f32
    tensor %in1: f32
    tensor %out: f32

    axis @a extent 8   strides { in0: 24576, in1: 0,     out: 1179648 }
    axis @b extent 64  strides { in0: 384,   in1: 1024,  out: 0 }
    axis @e extent 12  strides { in0: 0,     in1: 65536, out: 98304 }
    axis @f extent 256 strides { in0: 0,     in1: 4,     out: 384 }
    axis @g extent 96  strides { in0: 4,     in1: 0,     out: 4}

    primitive @zero : Zero axes { M: [@g], N: [@f] } metadata { data_type: f32 }
    primitive @contract : Contraction axes { M: [@g], N: [@f], K: [@b] } metadata { data_type: f32}

    schedule {
        roots: [@iter_a]

        iter @iter_a axis @a policy sequential children [@iter_b]
        iter @iter_e axis @e policy sequential children [@inv_zero, @inv_gemm]

        invoke @inv_zero primitive @zero
        invoke @inv_gemm primitive @gemm
    }
}
```

```
teir @v2_parallel {
    tensor %in0: f32
    tensor %in1: f32
    tensor %out: f32

    axis @a extent 8   strides { in0: 24576, in1: 0,     out: 1179648 }
    axis @b extent 64  strides { in0: 384,   in1: 1024,  out: 0 }
    axis @e extent 12  strides { in0: 0,     in1: 65536, out: 98304 }
    axis @f extent 256 strides { in0: 0,     in1: 4,     out: 384 }
    axis @g extent 96  strides { in0: 4,     in1: 0,     out: 4}

    primitive @zero : Zero axes { M: [@g], N: [@f] } metadata { data_type: f32 }
    primitive @contract : Contraction axes { M: [@g], N: [@f], K: [@b] } metadata { data_type: f32}

    schedule {
        roots: [@iter_a]

        iter @iter_a axis @a policy parallel children [@iter_e]
        iter @iter_e axis @e policy parallel children [@inv_zero, @inv_gemm]

        invoke @inv_zero primitive @zero
        invoke @inv_gemm primitive @gemm
    }
}
```

```
teir @v2_reordered {
    tensor %in0: f32
    tensor %in1: f32
    tensor %out: f32

    axis @a extent 8   strides { in0: 24576, in1: 0,     out: 1179648 }
    axis @b extent 64  strides { in0: 384,   in1: 1024,  out: 0 }
    axis @e extent 12  strides { in0: 0,     in1: 65536, out: 98304 }
    axis @f extent 256 strides { in0: 0,     in1: 4,     out: 384 }
    axis @g extent 96  strides { in0: 4,     in1: 0,     out: 4}

    primitive @zero : Zero axes { M: [@g], N: [@f] } metadata { data_type: f32 }
    primitive @contract : Contraction axes { M: [@g], N: [@f], K: [@b] } metadata { data_type: f32}

    schedule {
        roots: [@iter_e]

        iter @iter_e axis @e policy sequential children [@iter_a]
        iter @iter_a axis @a policy sequential children [@inv_zero, @inv_gemm]
        
        invoke @inv_zero primitive @zero
        invoke @inv_gemm primitive @gemm
    }
}
```



















## Build instructions

```
cd build
cmake ..
cmake --build .
```

```
/opt/homebrew/opt/cmake/bin/cmake -D CMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" ..
/opt/homebrew/opt/cmake/bin/cmake --build .
```












