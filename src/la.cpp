// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "la.hpp"
#include "algorithms.hpp"
#include <CL/sycl.hpp>

using namespace dolfinx;

namespace {
// //--------------------------------------------------------------------------
// inline void swap(std::int32_t* a, std::int32_t* b) {
//   std::int32_t t = *a;
//   *a = *b;
//   *b = t;
// }
//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix create_csr(cl::sycl::queue& queue,
                                             const experimental::sycl::memory::form_data_t& data) {
  std::int32_t ndofs = data.ndofs;
  std::int32_t ncells = data.ncells;
  std::int32_t ndofs_cell = data.dofs.get_range()[1];
  std::int32_t stored_nz = ncells * ndofs_cell * ndofs_cell;

  cl::sycl::range<1> cell_range(ncells);

  auto counter = cl::sycl::malloc_device<std::int32_t>(ndofs, queue);
  queue.fill(counter, 0, ndofs).wait();

  // Allocate device memory
  auto row_ptr = cl::sycl::malloc_device<std::int32_t>(ndofs + 1, queue);
  auto indices = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);
  cl::sycl::buffer<std::int32_t, 2> dofs_ = data.dofs;

  cl::sycl::event e0 = queue.submit([&](cl::sycl::handler& h) {
    cl::sycl::accessor dofs{dofs_, h, cl::sycl::read_only};
    h.parallel_for<class CountRowNz>(cell_range, [=](cl::sycl::id<1> Id) {
      int i = Id.get(0);
      for (int j = 0; j < ndofs_cell; j++) {
        for (int k = 0; k < ndofs_cell; k++) {
          std::int32_t row = dofs[i][j];
          auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[row]);
          cl::sycl::atomic<std::int32_t> counter{global_ptr};
          counter.fetch_add(1);
        }
      }
    });
  });

  queue.wait();

  // TODO: Improve exclusive scan implementation for GPUs
  experimental::sycl::algorithms::exclusive_scan(queue, counter, row_ptr, ndofs);

  queue.fill(counter, 0, ndofs).wait();
  cl::sycl::event e1 = queue.submit([&](cl::sycl::handler& h) {
    cl::sycl::accessor dofs{dofs_, h, cl::sycl::read_only};
    h.parallel_for<class ColIndices>(cell_range, [=](cl::sycl::id<1> id) {
      int i = id.get(0);
      for (int j = 0; j < ndofs_cell; j++) {
        std::int32_t row = dofs[i][j];
        auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[row]);
        cl::sycl::atomic<std::int32_t> state{global_ptr};
        for (int k = 0; k < ndofs_cell; k++) {
          std::int32_t current_count = state.fetch_add(1);
          std::int32_t pos = current_count + row_ptr[row];
          indices[pos] = dofs[i][k];
        }
      }
    });
  });

  queue.wait();

  experimental::sycl::la::CsrMatrix matrix{};
  matrix.indices = indices;
  matrix.indptr = row_ptr;
  matrix.nrows = ndofs;
  matrix.nnz = stored_nz;

  cl::sycl::free(counter, queue);

  return matrix;
}
//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix csr_remove_duplicate(cl::sycl::queue& queue,
                                                       experimental::sycl::la::CsrMatrix mat) {

  auto counter = cl::sycl::malloc_device<std::int32_t>(mat.nrows, queue);
  auto aux = cl::sycl::malloc_device<std::int32_t>(mat.nnz, queue);
  queue.fill(counter, 0, mat.nrows).wait();

  queue.parallel_for<class SortIndices>(cl::sycl::range<1>(mat.nrows), [=](cl::sycl::id<1> it) {
    int i = it.get(0);

    // // TODO: Improve performance of sorting algorithm
    // for (int j = mat.indptr[i]; j < mat.indptr[i + 1]; j++)
    //   for (int k = mat.indptr[i]; k < j; k++)
    //     if (mat.indices[k] > mat.indices[j])
    //       std::swap(mat.indices[k], mat.indices[j]);

    std::int32_t n = mat.indptr[i + 1] - mat.indptr[i];
    std::int32_t pos = mat.indptr[i];
    experimental::sycl::algorithms::radix_sort(&mat.indices[pos], &aux[pos], n);

    // Count number of unique column indices per row
    std::int32_t temp = -1;
    for (int j = mat.indptr[i]; j < mat.indptr[i + 1]; j++) {
      if (temp != mat.indices[j]) {
        counter[i]++;
        temp = mat.indices[j];
      }
    }
  });
  queue.wait();

  // create new csr matrix and remove repeated indices
  experimental::sycl::la::CsrMatrix out;
  out.indptr = cl::sycl::malloc_device<std::int32_t>(mat.nrows + 1, queue);
  experimental::sycl::algorithms::exclusive_scan(queue, counter, out.indptr, mat.nrows);
  out.nrows = mat.nrows;

  // number of nonzeros cannot be acessed directly on the host, instead use memcpy
  queue.memcpy(&out.nnz, &out.indptr[out.nrows], sizeof(std::int32_t)).wait();
  out.indices = cl::sycl::malloc_device<std::int32_t>(out.nnz, queue);
  out.data = cl::sycl::malloc_shared<double>(out.nnz, queue);

  queue.parallel_for<class UniqueIndices>(cl::sycl::range<1>(out.nrows), [=](cl::sycl::id<1> it) {
    int i = it.get(0);

    std::int32_t temp = -1;
    std::int32_t counter = 0;
    for (int j = mat.indptr[i]; j < mat.indptr[i + 1]; j++) {
      if (temp != mat.indices[j]) {
        temp = mat.indices[j];
        out.indices[out.indptr[i] + counter] = mat.indices[j];
        counter++;
      }
    }
  });
  queue.wait();

  return out;
}
//--------------------------------------------------------------------------
void free_csr(cl::sycl::queue& queue, experimental::sycl::la::CsrMatrix& mat) {
  queue.wait();
  cl::sycl::free(mat.indices, queue);
  cl::sycl::free(mat.indptr, queue);
}
} // namespace

//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix
experimental::sycl::la::create_csr_matrix(MPI_Comm comm, cl::sycl::queue& queue,
                                          const experimental::sycl::memory::form_data_t& data) {
  dolfinx::common::Timer t0("x Create CSR matrix");

  // Create csr matrix, with possible duplicate entries
  dolfinx::common::Timer t1("xx Create CSR matrix - extended");
  CsrMatrix csr_mat = create_csr(queue, data);
  t1.stop();

  // Remove duplicates and reorder columns
  dolfinx::common::Timer t2("xx Create CSR matrix - Remove duplicates");
  CsrMatrix new_matrix = csr_remove_duplicate(queue, csr_mat);
  queue.fill<double>(new_matrix.data, 0., new_matrix.nnz).wait();
  t2.stop();

  t0.stop();

  free_csr(queue, csr_mat);

  return new_matrix;
}