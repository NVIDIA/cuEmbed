// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// clang-format on

//! \file
#ifndef CUEMBED_INCLUDE_EMBEDDING_LOOKUP_TYPES_CUH_
#define CUEMBED_INCLUDE_EMBEDDING_LOOKUP_TYPES_CUH_

#include <cuda_fp16.h>

namespace cuembed {

// TODO(zejiaz): support more modes in the kernels
enum class CombineMode { kSum, kMean, kConcat };

// Various vectorized types and support functions.
typedef struct __align__(32) {
  float a, b, c, d, e, f, g, h;
}
float8;

typedef struct __align__(16) {
  __half a, b, c, d, e, f, g, h;
}
half8;

typedef struct __align__(8) {
  __half x, y, z, w;
}
half4;

// VecCast functions for float <=> half, float2 <=> half2, float4 <=> half4, and
// float8 <=> half8.
// Default is to perform no cast. This should only be used when ToType equals
// to FromType.
template <typename ToType, typename FromType>
__device__ __host__ __forceinline__ ToType VecCast(const FromType& value) {
  return value;
}

// Specialization for casting from float to half
template <>
__device__ __host__ __forceinline__ half
VecCast<half, float>(const float& value) {
  return __float2half(value);
}

// Specialization for casting from half to float
template <>
__device__ __host__ __forceinline__ float VecCast<float, half>(
    const half& value) {
  return __half2float(value);
}

// Specialization for casting from float2 to half2
template <>
__device__ __host__ __forceinline__ half2
VecCast<half2, float2>(const float2& value) {
  half2 result;
  result.x = __float2half(value.x);
  result.y = __float2half(value.y);
  return result;
}

// Specialization for casting from half2 to float2
template <>
__device__ __host__ __forceinline__ float2
VecCast<float2, half2>(const half2& value) {
  float2 result;
  result.x = __half2float(value.x);
  result.y = __half2float(value.y);
  return result;
}

// Specialization for casting from float4 to half4
template <>
__device__ __host__ __forceinline__ half4
VecCast<half4, float4>(const float4& value) {
  half4 result;
  result.x = __float2half(value.x);
  result.y = __float2half(value.y);
  result.z = __float2half(value.z);
  result.w = __float2half(value.w);
  return result;
}

// Specialization for casting from half4 to float4
template <>
__device__ __host__ __forceinline__ float4
VecCast<float4, half4>(const half4& value) {
  float4 result;
  result.x = __half2float(value.x);
  result.y = __half2float(value.y);
  result.z = __half2float(value.z);
  result.w = __half2float(value.w);
  return result;
}

// Specialization for casting from float4 to half4
template <>
__device__ __host__ __forceinline__ half8
VecCast<half8, float8>(const float8& value) {
  half8 result;
  result.a = __float2half(value.a);
  result.b = __float2half(value.b);
  result.c = __float2half(value.c);
  result.d = __float2half(value.d);
  result.e = __float2half(value.e);
  result.f = __float2half(value.f);
  result.g = __float2half(value.g);
  result.h = __float2half(value.h);
  return result;
}

// Specialization for casting from half4 to float4
template <>
__device__ __host__ __forceinline__ float8
VecCast<float8, half8>(const half8& value) {
  float8 result;
  result.a = __half2float(value.a);
  result.b = __half2float(value.b);
  result.c = __half2float(value.c);
  result.d = __half2float(value.d);
  result.e = __half2float(value.e);
  result.f = __half2float(value.f);
  result.g = __half2float(value.g);
  result.h = __half2float(value.h);
  return result;
}

__device__ __host__ __forceinline__ void operator+=(
    half8& lhs, half8 rhs) {  // NOLINT(runtime/references)
  lhs.a += rhs.a;
  lhs.b += rhs.b;
  lhs.c += rhs.c;
  lhs.d += rhs.d;
  lhs.e += rhs.e;
  lhs.f += rhs.f;
  lhs.g += rhs.g;
  lhs.h += rhs.h;
}

__device__ __host__ __forceinline__ void operator+=(
    float8& lhs, const float8& rhs) {  // NOLINT(runtime/references)
  lhs.a += rhs.a;
  lhs.b += rhs.b;
  lhs.c += rhs.c;
  lhs.d += rhs.d;
  lhs.e += rhs.e;
  lhs.f += rhs.f;
  lhs.g += rhs.g;
  lhs.h += rhs.h;
}

__device__ __host__ __forceinline__ void operator+=(
    float4& lhs, const float4& rhs) {  // NOLINT(runtime/references)
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  lhs.w += rhs.w;
}

__device__ __host__ __forceinline__ void operator+=(
    float2& lhs, const float2& rhs) {  // NOLINT(runtime/references)
  lhs.x += rhs.x;
  lhs.y += rhs.y;
}

__device__ __host__ __forceinline__ void operator+=(
    half4& lhs, half4 rhs) {  // NOLINT(runtime/references)
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  lhs.w += rhs.w;
}

__device__ __host__ __forceinline__ void operator+=(
    float& lhs, const half& rhs) {  // NOLINT(runtime/references)
  lhs += __half2float(rhs);
}

__device__ __host__ __forceinline__ void operator+=(
    float2& lhs, const half2& rhs) {  // NOLINT(runtime/references)
  lhs.x += __half2float(rhs.x);
  lhs.y += __half2float(rhs.y);
}

__device__ __host__ __forceinline__ void operator+=(
    float4& lhs, half4 rhs) {  // NOLINT(runtime/references)
  lhs.x += __half2float(rhs.x);
  lhs.y += __half2float(rhs.y);
  lhs.z += __half2float(rhs.z);
  lhs.w += __half2float(rhs.w);
}

__device__ __host__ __forceinline__ void operator+=(
    float8& lhs, const half8& rhs) {  // NOLINT(runtime/references)
  lhs.a += __half2float(rhs.a);
  lhs.b += __half2float(rhs.b);
  lhs.c += __half2float(rhs.c);
  lhs.d += __half2float(rhs.d);
  lhs.e += __half2float(rhs.e);
  lhs.f += __half2float(rhs.f);
  lhs.g += __half2float(rhs.g);
  lhs.h += __half2float(rhs.h);
}

__device__ __host__ __forceinline__ float8
operator*(const float8& lhs, const float rhs) {  // NOLINT(runtime/references)
  float8 result;
  result.a = lhs.a * rhs;
  result.b = lhs.b * rhs;
  result.c = lhs.c * rhs;
  result.d = lhs.d * rhs;
  result.e = lhs.e * rhs;
  result.f = lhs.f * rhs;
  result.g = lhs.g * rhs;
  result.h = lhs.h * rhs;
  return result;
}

__device__ __host__ __forceinline__ float4
operator*(const float4& lhs, const float rhs) {  // NOLINT(runtime/references)
  float4 result;
  result.x = lhs.x * rhs;
  result.y = lhs.y * rhs;
  result.z = lhs.z * rhs;
  result.w = lhs.w * rhs;
  return result;
}

__device__ __host__ __forceinline__ float2
operator*(const float2& lhs, const float rhs) {  // NOLINT(runtime/references)
  float2 result;
  result.x = lhs.x * rhs;
  result.y = lhs.y * rhs;
  return result;
}

__device__ __host__ __forceinline__ float8
operator*(const float8& lhs, const __half rhs) {  // NOLINT(runtime/references)
  return lhs * __half2float(rhs);
}

__device__ __host__ __forceinline__ float4
operator*(const float4& lhs, const __half rhs) {  // NOLINT(runtime/references)
  return lhs * __half2float(rhs);
}

__device__ __host__ __forceinline__ float2
operator*(const float2& lhs, const __half rhs) {  // NOLINT(runtime/references)
  return lhs * __half2float(rhs);
}

__device__ __host__ __forceinline__ float operator*(
    const float& lhs, const __half rhs) {  // NOLINT(runtime/references)
  return lhs * __half2float(rhs);
}

__device__ __host__ __forceinline__ half8
operator*(const half8& lhs, const __half rhs) {  // NOLINT(runtime/references)
  half8 result;
  result.a = lhs.a * rhs;
  result.b = lhs.b * rhs;
  result.c = lhs.c * rhs;
  result.d = lhs.d * rhs;
  result.e = lhs.e * rhs;
  result.f = lhs.f * rhs;
  result.g = lhs.g * rhs;
  result.h = lhs.h * rhs;
  return result;
}

__device__ __host__ __forceinline__ half4
operator*(const half4& lhs, const __half rhs) {  // NOLINT(runtime/references)
  half4 result;
  result.x = lhs.x * rhs;
  result.y = lhs.y * rhs;
  result.z = lhs.z * rhs;
  result.w = lhs.w * rhs;
  return result;
}

__device__ __host__ __forceinline__ half2
operator*(const half2& lhs, const __half rhs) {  // NOLINT(runtime/references)
  half2 result;
  result.x = lhs.x * rhs;
  result.y = lhs.y * rhs;
  return result;
}

__device__ __host__ __forceinline__ half8
operator*(const half8& lhs, const float rhs) {  // NOLINT(runtime/references)
  return lhs * __float2half(rhs);
}

__device__ __host__ __forceinline__ half4
operator*(const half4& lhs, const float rhs) {  // NOLINT(runtime/references)
  return lhs * __float2half(rhs);
}

__device__ __host__ __forceinline__ half2
operator*(const half2& lhs, const float rhs) {  // NOLINT(runtime/references)
  return lhs * __float2half(rhs);
}

__device__ __host__ __forceinline__ half
operator*(const half& lhs, const float rhs) {  // NOLINT(runtime/references)
  return lhs * __float2half(rhs);
}

// Atomic functions for vector types
template <typename VecType>
__device__ __forceinline__ VecType VecAtomicAdd(VecType* addr,
                                                const VecType& value) {
  return atomicAdd(addr, value);
}

#if __CUDA_ARCH__ >= 900

// Specialization for atomic add of half4, half8, and float8
template <>
__device__ __forceinline__ half4 VecAtomicAdd<half4>(half4* addr,
                                                     const half4& value) {
  half2 value0;
  half2 value1;
  value0.x = value.x;
  value0.y = value.y;
  value1.x = value.z;
  value1.y = value.w;
  half2 res0;
  half2 res1;
  res0 = atomicAdd(reinterpret_cast<half2*>(addr), value0);
  res1 = atomicAdd(reinterpret_cast<half2*>(addr) + 1, value1);
  half4 res;
  res.x = res0.x;
  res.y = res0.y;
  res.z = res1.x;
  res.w = res1.y;
  return res;
}

template <>
__device__ __forceinline__ half8 VecAtomicAdd<half8>(half8* addr,
                                                     const half8& value) {
  half2 value0;
  half2 value1;
  half2 value2;
  half2 value3;
  value0.x = value.a;
  value0.y = value.b;
  value1.x = value.c;
  value1.y = value.d;
  value2.x = value.e;
  value2.y = value.f;
  value3.x = value.g;
  value3.y = value.h;
  half2 res0;
  half2 res1;
  half2 res2;
  half2 res3;
  res0 = atomicAdd(reinterpret_cast<half2*>(addr), value0);
  res1 = atomicAdd(reinterpret_cast<half2*>(addr) + 1, value1);
  res2 = atomicAdd(reinterpret_cast<half2*>(addr) + 2, value2);
  res3 = atomicAdd(reinterpret_cast<half2*>(addr) + 3, value3);
  half8 res;
  res.a = res0.x;
  res.b = res0.y;
  res.c = res1.x;
  res.d = res1.y;
  res.e = res2.x;
  res.f = res2.y;
  res.g = res3.x;
  res.h = res3.y;
  return res;
}

template <>
__device__ __forceinline__ float8 VecAtomicAdd<float8>(float8* addr,
                                                       const float8& value) {
  float4 value0;
  float4 value1;
  value0.x = value.a;
  value0.y = value.b;
  value0.z = value.c;
  value0.w = value.d;
  value1.x = value.e;
  value1.y = value.f;
  value1.z = value.g;
  value1.w = value.h;
  float4 res0 = atomicAdd(reinterpret_cast<float4*>(addr), value0);
  float4 res1 = atomicAdd(reinterpret_cast<float4*>(addr) + 1, value1);
  float8 res;
  res.a = res0.x;
  res.b = res0.y;
  res.c = res0.z;
  res.d = res0.w;
  res.e = res1.x;
  res.f = res1.y;
  res.g = res1.z;
  res.h = res1.w;
  return res;
}

#else  // CUDA_ARCH < 900

// Specialization for atomic add of half2, float2, half4, float4, half8, float8
template <>
__device__ __forceinline__ half2 VecAtomicAdd<half2>(half2* addr,
                                                     const half2& value) {
  half2 res;
  res.x = atomicAdd(reinterpret_cast<half*>(addr), value.x);
  res.y = atomicAdd(reinterpret_cast<half*>(addr) + 1, value.y);
  return res;
}

template <>
__device__ __forceinline__ float2 VecAtomicAdd<float2>(float2* addr,
                                                       const float2& value) {
  float2 res;
  res.x = atomicAdd(reinterpret_cast<float*>(addr), value.x);
  res.y = atomicAdd(reinterpret_cast<float*>(addr) + 1, value.y);
  return res;
}

template <>
__device__ __forceinline__ half4 VecAtomicAdd<half4>(half4* addr,
                                                     const half4& value) {
  half4 res;
  res.x = atomicAdd(reinterpret_cast<half*>(addr), value.x);
  res.y = atomicAdd(reinterpret_cast<half*>(addr) + 1, value.y);
  res.z = atomicAdd(reinterpret_cast<half*>(addr) + 2, value.z);
  res.w = atomicAdd(reinterpret_cast<half*>(addr) + 3, value.w);
  return res;
}

template <>
__device__ __forceinline__ float4 VecAtomicAdd<float4>(float4* addr,
                                                       const float4& value) {
  float4 res;
  res.x = atomicAdd(reinterpret_cast<float*>(addr), value.x);
  res.y = atomicAdd(reinterpret_cast<float*>(addr) + 1, value.y);
  res.z = atomicAdd(reinterpret_cast<float*>(addr) + 2, value.z);
  res.w = atomicAdd(reinterpret_cast<float*>(addr) + 3, value.w);
  return res;
}

template <>
__device__ __forceinline__ half8 VecAtomicAdd<half8>(half8* addr,
                                                     const half8& value) {
  half8 res;
  res.a = atomicAdd(reinterpret_cast<half*>(addr), value.a);
  res.b = atomicAdd(reinterpret_cast<half*>(addr) + 1, value.b);
  res.c = atomicAdd(reinterpret_cast<half*>(addr) + 2, value.c);
  res.d = atomicAdd(reinterpret_cast<half*>(addr) + 3, value.d);
  res.e = atomicAdd(reinterpret_cast<half*>(addr) + 4, value.e);
  res.f = atomicAdd(reinterpret_cast<half*>(addr) + 5, value.f);
  res.g = atomicAdd(reinterpret_cast<half*>(addr) + 6, value.g);
  res.h = atomicAdd(reinterpret_cast<half*>(addr) + 7, value.h);
  return res;
}

template <>
__device__ __forceinline__ float8 VecAtomicAdd<float8>(float8* addr,
                                                       const float8& value) {
  float8 res;
  res.a = atomicAdd(reinterpret_cast<float*>(addr), value.a);
  res.b = atomicAdd(reinterpret_cast<float*>(addr) + 1, value.b);
  res.c = atomicAdd(reinterpret_cast<float*>(addr) + 2, value.c);
  res.d = atomicAdd(reinterpret_cast<float*>(addr) + 3, value.d);
  res.e = atomicAdd(reinterpret_cast<float*>(addr) + 4, value.e);
  res.f = atomicAdd(reinterpret_cast<float*>(addr) + 5, value.f);
  res.g = atomicAdd(reinterpret_cast<float*>(addr) + 6, value.g);
  res.h = atomicAdd(reinterpret_cast<float*>(addr) + 7, value.h);
  return res;
}

#endif

// A helper struct used to construct vectorized types using type
// specialization.
// Example usage:
//   int elem_per_load = 2;
//   bool fp16_math = false;
//   typedef typename VecTypeHelper<float, elem_per_load, fp16_math>::LoadType
//   LoadType;
//   LaunchKernel<LoadType>(...); // <- This launches kernel with float2.
template <typename T, int count>
struct VecTypeBase;

template <typename T, int count, bool fp16_math>
struct VecTypeHelper : public VecTypeBase<T, count> {
  // Inherits LoadType and ReduceType from VecTypeBase
};

// float8 is used only for compilation completeness.
template <>
struct VecTypeBase<float, 8> {
  using LoadType = float8;
  using ReduceType = float8;
};

template <>
struct VecTypeBase<float, 4> {
  using LoadType = float4;
  using ReduceType = float4;
};

template <>
struct VecTypeBase<float, 2> {
  using LoadType = float2;
  using ReduceType = float2;
};

template <>
struct VecTypeBase<float, 1> {
  using LoadType = float;
  using ReduceType = float;
};

template <>
struct VecTypeBase<__half, 8> {
  using LoadType = half8;
  using ReduceType = half8;
};
template <>
struct VecTypeBase<__half, 4> {
  using LoadType = half4;
  using ReduceType = half4;
};

template <>
struct VecTypeBase<__half, 2> {
  using LoadType = __half2;
  using ReduceType = __half2;
};

template <>
struct VecTypeBase<__half, 1> {
  using LoadType = __half;
  using ReduceType = __half;
};

template <>
struct VecTypeHelper<__half, 8, false> {
  using LoadType = half8;
  using ReduceType = float8;
};
template <>
struct VecTypeHelper<__half, 4, false> {
  using LoadType = half4;
  using ReduceType = float4;
};

template <>
struct VecTypeHelper<__half, 2, false> {
  using LoadType = __half2;
  using ReduceType = float2;
};

template <>
struct VecTypeHelper<__half, 1, false> {
  using LoadType = __half;
  using ReduceType = float;
};

// Default behavior of unwrapping element type (e.g., float) from a struct
// (e.g., float with cache)
// User who needs more complicated behavior should provide their own template
// specialization of this struct.
template <typename T>
struct GetElemType {
  using Type = T;
};

template <typename T>
using GetElemT = typename GetElemType<T>::Type;

}  // namespace cuembed

#endif  // CUEMBED_INCLUDE_EMBEDDING_LOOKUP_TYPES_CUH_
