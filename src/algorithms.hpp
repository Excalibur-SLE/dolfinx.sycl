// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>

#include <numeric>
#include <vector>

namespace dolfinx::experimental::sycl::algorithms
{
//--------------------------------------------------------------------------
// A utility function to swap two elements
template <typename T>
void swap(T* a, T* b)
{
  T t = *a;
  *a = *b;
  *b = t;
}
//--------------------------------------------------------------------------
void exclusive_scan(cl::sycl::queue& queue, std::int32_t* input,
                    std::int32_t* output, std::int32_t size)
{
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
int binary_search(int* arr, int first, int last, int x)
{
  while (first <= last)
  {
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
//--------------------------------------------------------------------------
} // namespace dolfinx::experimental::sycl::algorithms