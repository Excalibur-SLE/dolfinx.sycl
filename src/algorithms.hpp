// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>

#include <numeric>
#include <vector>

namespace dolfinx::experimental::sycl::algorithms {
//--------------------------------------------------------------------------
void exclusive_scan(cl::sycl::queue& queue, std::int32_t* input, std::int32_t* output,
                    std::int32_t size) {
  common::Timer t0("xxx Exclusive Scan");
  // FIXME: do not copy data back to host!!!!!!!!!!!!!!
  std::vector<std::int32_t> in(size, 0);
  std::vector<std::int32_t> out(size + 1, 0);

  queue.memcpy(in.data(), input, size * sizeof(std::int32_t)).wait();

  std::partial_sum(in.begin(), in.end(), out.begin() + 1);

  queue.memcpy(output, out.data(), (size + 1) * sizeof(std::int32_t)).wait();

  t0.stop();
}
//--------------------------------------------------------------------------
int binary_search(int* arr, int first, int last, int x) {
  while (first <= last) {
    int middle = first + (last - first) / 2;

    if (arr[middle] == x)
      return middle;

    if (arr[middle] < x)
      first = middle + 1;
    else
      last = middle - 1;
  }

  return -1;
}

template <typename ValueType>
void radix_sort(ValueType* values, ValueType* valuesAux, std::int32_t n) {
  if (n <= 1)
    return;
  ValueType maxVal = 0;
  for (std::int32_t i = 0; i < n; i++) {
    if (maxVal < values[i])
      maxVal = values[i];
  }
  // determine how many significant bits the data has
  int passes = 0;
  while (maxVal) {
    maxVal >>= 4;
    passes++;
  }
  // Is the data currently held in values (false) or valuesAux (true)?
  bool inAux = false;
  // sort 4 bits at a time, into 16 buckets
  ValueType mask = 0xF;
  // maskPos counts the low bit index of mask (0, 4, 8, ...)
  std::int32_t maskPos = 0;
  for (int p = 0; p < passes; p++) {
    // Count the number of elements in each bucket
    std::int32_t count[16] = {0};
    std::int32_t offset[17];
    if (!inAux) {
      for (std::int32_t i = 0; i < n; i++) {
        count[(values[i] & mask) >> maskPos]++;
      }
    } else {
      for (std::int32_t i = 0; i < n; i++) {
        count[(valuesAux[i] & mask) >> maskPos]++;
      }
    }
    offset[0] = 0;
    // get offset as the prefix sum for count
    for (std::int32_t i = 0; i < 16; i++) {
      offset[i + 1] = offset[i] + count[i];
    }
    // now for each element in [lo, hi), move it to its offset in the other buffer
    // this branch should be ok because whichBuf is the same on all threads
    if (!inAux) {
      for (std::int32_t i = 0; i < n; i++) {
        std::int32_t bucket = (values[i] & mask) >> maskPos;
        valuesAux[offset[bucket + 1] - count[bucket]] = values[i];
        count[bucket]--;
      }
    } else {
      for (std::int32_t i = 0; i < n; i++) {
        std::int32_t bucket = (valuesAux[i] & mask) >> maskPos;
        values[offset[bucket + 1] - count[bucket]] = valuesAux[i];
        count[bucket]--;
      }
    }
    inAux = !inAux;
    mask = mask << 4;
    maskPos += 4;
  }
  // Move values back into main array if they are currently in aux.
  // This is the case if an odd number of rounds were done.
  if (inAux) {
    for (std::int32_t i = 0; i < n; i++) {
      values[i] = valuesAux[i];
    }
  }
}

//--------------------------------------------------------------------------
} // namespace dolfinx::experimental::sycl::algorithms