// Copyright (C) 2020 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>

#ifdef SYCL_DEVICE_ONLY
#undef SYCL_DEVICE_ONLY
#include <dolfinx.h>
#define SYCL_DEVICE_ONLY
#else
#include <dolfinx.h>
#endif

#include <mpi.h>

#include <cstdint>

#include "memory.hpp"

using namespace dolfinx;

namespace dolfinx::experimental::sycl::la
{

struct CsrMatrix
{
  double* data;
  std::int32_t* indptr;
  std::int32_t* indices;

  std::int32_t nrows;
  std::int32_t nnz;
};

struct AdjacencyList
{
  std::int32_t* indptr;
  std::int32_t* indices;
  std::int32_t num_nodes;
  std::int32_t num_links;
};

/// Create a CSR matrix
/// @param[in] comm Form data
/// @param[in] queue Form data
/// @param[in] data Form data
CsrMatrix
create_csr_matrix(MPI_Comm comm, cl::sycl::queue& queue,
                  const experimental::sycl::memory::form_data_t& data);

int32_t*
compute_lookup_table(cl::sycl::queue& queue,
                     const experimental::sycl::la::CsrMatrix& mat,
                     const experimental::sycl::memory::form_data_t& data);

AdjacencyList
compute_matrix_acc_map(cl::sycl::queue& queue,
                       const experimental::sycl::la::CsrMatrix& mat,
                       const experimental::sycl::memory::form_data_t& data);

AdjacencyList
compute_vector_acc_map(MPI_Comm comm, cl::sycl::queue& queue,
                       const experimental::sycl::memory::form_data_t& data);

} // namespace dolfinx::experimental::sycl::la