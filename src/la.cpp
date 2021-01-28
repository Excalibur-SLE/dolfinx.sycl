// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "la.hpp"
#include "algorithms.hpp"

using namespace dolfinx;

namespace
{
//--------------------------------------------------------------------------
[[maybe_unused]] inline void swap(std::int32_t* a, std::int32_t* b)
{
  std::int32_t t = *a;
  *a = *b;
  *b = t;
}
//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix
create_csr(cl::sycl::queue& queue,
           const experimental::sycl::memory::form_data_t& data)
{
  std::int32_t ndofs = data.ndofs;
  std::int32_t ncells = data.ncells;
  std::int32_t ndofs_cell = data.ndofs_cell;
  std::int32_t stored_nz = ncells * ndofs_cell * ndofs_cell;

  auto counter = cl::sycl::malloc_device<std::int32_t>(ndofs, queue);
  queue.fill(counter, 0, ndofs).wait();

  // Allocate device memory
  auto row_ptr = cl::sycl::malloc_device<std::int32_t>(ndofs + 1, queue);
  auto indices = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);

  // Count the number of stored nonzeros per row
  auto count_nonzeros = [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    std::int32_t offset = i * ndofs_cell;
    for (int j = 0; j < ndofs_cell; j++)
    {
      for (int k = 0; k < ndofs_cell; k++)
      {
        std::int32_t row = data.dofs[offset + j];
        auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[row]);
        cl::sycl::atomic<std::int32_t> counter{global_ptr};
        counter.fetch_add(1);
      }
    }
  };

  // Submit kernel to the queue
  cl::sycl::range<1> cell_range(ncells);
  queue.parallel_for<class CountRowNz>(cell_range, count_nonzeros).wait();

  // TODO: Improve exclusive scan implementation for GPUs
  experimental::sycl::algorithms::exclusive_scan(queue, counter, row_ptr,
                                                 ndofs);

  // Populate column indices, might have repeated values
  auto populate_indices = [=](cl::sycl::id<1> id) {
    int i = id.get(0);

    std::int32_t offset = i * ndofs_cell;
    for (int j = 0; j < ndofs_cell; j++)
    {
      std::int32_t row = data.dofs[offset + j];
      auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[row]);
      cl::sycl::atomic<std::int32_t> state{global_ptr};
      for (int k = 0; k < ndofs_cell; k++)
      {
        std::int32_t current_count = state.fetch_add(1);
        std::int32_t pos = current_count + row_ptr[row];
        indices[pos] = data.dofs[offset + k];
      }
    }
  };

  queue.fill(counter, 0, ndofs).wait();
  queue.parallel_for<class ColIndices>(cell_range, populate_indices).wait();

  experimental::sycl::la::CsrMatrix matrix{};
  matrix.indices = indices;
  matrix.indptr = row_ptr;
  matrix.nrows = ndofs;
  matrix.nnz = stored_nz;

  cl::sycl::free(counter, queue);

  return matrix;
}
//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix
csr_remove_duplicate(cl::sycl::queue& queue,
                     experimental::sycl::la::CsrMatrix mat)
{

  std::int32_t nrows = mat.nrows;
  std::int32_t* row_ptr = mat.indptr;
  std::int32_t* indices = mat.indices;

  auto counter = cl::sycl::malloc_device<std::int32_t>(nrows, queue);
  queue.fill(counter, 0, nrows).wait();

  queue.parallel_for<class SortIndices>(
      cl::sycl::range<1>(nrows), [=](cl::sycl::id<1> it) {
        int i = it.get(0);

        std::int32_t begin = row_ptr[i];
        std::int32_t end = row_ptr[i + 1];
        std::int32_t size = end - begin;

#ifdef __LLVM_SYCL__
        // TODO: Improve performance of sorting algorithm
        for (std::int32_t j = 0; j < size - 1; j++)
          for (std::int32_t k = 0; k < size - j - 1; k++)
            if (indices[begin + k] > indices[begin + k + 1])
              swap(&indices[begin + k], &indices[begin + k + 1]);
#else
      std::sort(indices + begin, indices + end);
#endif
        // Count number of unique column indices per row
        std::int32_t temp = -1;
        for (std::int32_t j = 0; j < size; j++)
        {
          if (temp != indices[begin + j])
          {
            counter[i]++;
            temp = indices[begin + j];
          }
        }
      });
  queue.wait();

  // create new csr matrix and remove repeated indices
  experimental::sycl::la::CsrMatrix out;
  out.indptr = cl::sycl::malloc_device<std::int32_t>(nrows + 1, queue);
  experimental::sycl::algorithms::exclusive_scan(queue, counter, out.indptr,
                                                 nrows);
  out.nrows = mat.nrows;

  // number of nonzeros cannot be acessed directly on the host, instead use
  // memcpy
  queue.memcpy(&out.nnz, &out.indptr[nrows], sizeof(std::int32_t)).wait();
  out.indices = cl::sycl::malloc_device<std::int32_t>(out.nnz, queue);
  out.data = cl::sycl::malloca_device<double>(out.nnz, queue);

  queue.parallel_for<class UniqueIndices>(
      cl::sycl::range<1>(nrows), [=](cl::sycl::id<1> it) {
        int i = it.get(0);

        // old rowsize
        std::int32_t row_size = row_ptr[i + 1] - row_ptr[i];

        std::int32_t temp = -1;
        std::int32_t counter = 0;
        for (std::int32_t j = 0; j < row_size; j++)
        {
          if (temp != indices[row_ptr[i] + j])
          {
            temp = indices[row_ptr[i] + j];
            out.indices[out.indptr[i] + counter] = indices[row_ptr[i] + j];
            counter++;
          }
        }
      });
  queue.wait();

  return out;
}
//--------------------------------------------------------------------------
void free_csr(cl::sycl::queue& queue, experimental::sycl::la::CsrMatrix& mat)
{
  queue.wait();
  cl::sycl::free(mat.indices, queue);
  cl::sycl::free(mat.indptr, queue);
}

} // namespace

//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix experimental::sycl::la::create_csr_matrix(
    MPI_Comm comm, cl::sycl::queue& queue,
    const experimental::sycl::memory::form_data_t& data)
{
  dolfinx::common::Timer t0("x Create CSR matrix");

  // Create csr matrix, with possible duplicate entries
  dolfinx::common::Timer t1("xx Create CSR matrix - extended");
  CsrMatrix csr_mat = create_csr(queue, data);
  t1.stop();

  // Remove duplicates and reorder columns
  dolfinx::common::Timer t2("xx Create CSR matrix - Remove duplicates");
  CsrMatrix new_matrix = csr_remove_duplicate(queue, csr_mat);
  free_csr(queue, csr_mat);
  t2.stop();

  t0.stop();

  return new_matrix;
}
//--------------------------------------------------------------------------
experimental::sycl::la::AdjacencyList
experimental::sycl::la::compute_matrix_acc_map(
    cl::sycl::queue& queue, const experimental::sycl::la::CsrMatrix& mat,
    const experimental::sycl::memory::form_data_t& data)
{

  experimental::sycl::la::AdjacencyList acc_map;
  acc_map.num_links = data.ndofs_cell * data.ndofs_cell * data.ncells;
  acc_map.num_nodes = mat.nnz;
  acc_map.indptr = cl::sycl::malloc_device<std::int32_t>(mat.nnz + 1, queue);
  acc_map.indices
      = cl::sycl::malloc_device<std::int32_t>(acc_map.num_links, queue);

  cl::sycl::range<1> range(data.ncells);

  auto counter = cl::sycl::malloc_device<std::int32_t>(mat.nnz, queue);
  queue.fill(counter, 0, mat.nnz).wait();

  queue.parallel_for<class AllocateDataAccMap>(range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    int dof_offset = data.ndofs_cell * i;
    for (int j = 0; j < data.ndofs_cell; j++)
    {
      int row = data.dofs[dof_offset + j];
      int first = mat.indptr[row];
      int last = mat.indptr[row + 1];
      for (int k = 0; k < data.ndofs_cell; k++)
      {
        int ind = data.dofs[dof_offset + k];
        int pos = experimental::sycl::algorithms::binary_search(
            mat.indices, first, last, ind);
        auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[pos]);
        cl::sycl::atomic<std::int32_t> state{global_ptr};
        state.fetch_add(1);
      }
    }
  });
  queue.wait_and_throw();

  // TODO: Improve exclusive scan implementation for GPUs
  experimental::sycl::algorithms::exclusive_scan(queue, counter, acc_map.indptr,
                                                 acc_map.num_nodes);
  queue.fill(counter, 0, mat.nnz).wait();

  queue.parallel_for<class AddDataAccMap>(range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    int dof_offset = data.ndofs_cell * i;
    int mat_offset = data.ndofs_cell * data.ndofs_cell * i;
    for (int j = 0; j < data.ndofs_cell; j++)
    {
      int row = data.dofs[dof_offset + j];
      int first = mat.indptr[row];
      int last = mat.indptr[row + 1];
      for (int k = 0; k < data.ndofs_cell; k++)
      {
        int ind = data.dofs[dof_offset + k];
        int pos = experimental::sycl::algorithms::binary_search(
            mat.indices, first, last, ind);
        auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[pos]);
        cl::sycl::atomic<std::int32_t> state{global_ptr};
        std::int32_t current_count = state.fetch_add(1);
        std::int32_t pos_list = current_count + acc_map.indptr[pos];
        acc_map.indices[pos_list] = mat_offset + j * data.ndofs_cell + k;
      }
    }
  });
  queue.wait_and_throw();

  return acc_map;
}
//--------------------------------------------------------------------------
int32_t* experimental::sycl::la::compute_lookup_table(
    cl::sycl::queue& queue, const experimental::sycl::la::CsrMatrix& mat,
    const experimental::sycl::memory::form_data_t& data)
{
  auto ext_nz = data.ndofs_cell * data.ndofs_cell * data.ncells;
  auto lookup = cl::sycl::malloc_device<std::int32_t>(ext_nz, queue);
  cl::sycl::range<1> range(data.ncells);
  queue.parallel_for<class LookupTable>(range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    int dof_offset = data.ndofs_cell * i;
    int mat_offset = data.ndofs_cell * data.ndofs_cell * i;
    for (int j = 0; j < data.ndofs_cell; j++)
    {
      int row = data.dofs[dof_offset + j];
      int first = mat.indptr[row];
      int last = mat.indptr[row + 1];
      for (int k = 0; k < data.ndofs_cell; k++)
      {
        int ind = data.dofs[dof_offset + k];
        int pos = experimental::sycl::algorithms::binary_search(
            mat.indices, first, last, ind);
        lookup[mat_offset + j * data.ndofs_cell + k] = pos;
      }
    }
  });
  queue.wait_and_throw();

  return lookup;
}
//--------------------------------------------------------------------------
experimental::sycl::la::AdjacencyList
experimental::sycl::la::compute_vector_acc_map(
    MPI_Comm comm, cl::sycl::queue& queue,
    const experimental::sycl::memory::form_data_t& data)
{

  auto counter = cl::sycl::malloc_device<std::int32_t>(data.ndofs, queue);
  queue.fill<std::int32_t>(counter, 0, data.ndofs).wait();

  // Count the number times the dof is shared
  cl::sycl::range<1> cell_range(data.ncells);
  queue.parallel_for<class CountSharedDofs>(
      cell_range, [=](cl::sycl::id<1> Id) {
        int i = Id.get(0);
        std::int32_t offset = i * data.ndofs_cell;
        for (int j = 0; j < data.ndofs_cell; j++)
        {
          std::int32_t dof = data.dofs[offset + j];
          auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[dof]);
          cl::sycl::atomic<std::int32_t> state{global_ptr};
          state.fetch_add(1);
        }
      });
  queue.wait();

  // Create accumulator adjacency list
  experimental::sycl::la::AdjacencyList acc;
  acc.num_nodes = data.ndofs;
  acc.num_links = data.ndofs_cell * data.ncells;
  acc.indptr = cl::sycl::malloc_device<std::int32_t>(acc.num_nodes + 1, queue);
  experimental::sycl::algorithms::exclusive_scan(queue, counter, acc.indptr,
                                                 acc.num_nodes);

  acc.indices = cl::sycl::malloc_device<std::int32_t>(acc.num_links, queue);
  queue.fill<std::int32_t>(counter, 0, data.ndofs).wait();

  // Position to accumulate
  queue.parallel_for<class GatherDofs>(cell_range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    std::int32_t offset = i * data.ndofs_cell;
    for (int j = 0; j < data.ndofs_cell; j++)
    {
      std::int32_t dof = data.dofs[offset + j];
      auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[dof]);
      cl::sycl::atomic<std::int32_t> state{global_ptr};
      std::int32_t current_count = state.fetch_add(1);

      std::int32_t pos = acc.indptr[dof] + current_count;
      acc.indices[pos] = offset + j;
    }
  });

  queue.wait();

  return acc;
}