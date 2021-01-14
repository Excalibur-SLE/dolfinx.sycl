// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#undef SYCL_DEVICE_ONLY
#include <dolfinx.h>
#define SYCL_DEVICE_ONLY

#include "assemble_impl.hpp"
#include "la.hpp"
#include "memory.hpp"

using namespace dolfinx::experimental::sycl;

namespace dolfinx::experimental::sycl::assemble
{
//--------------------------------------------------------------------------
// Submit vector assembly kernels to queue
double* assemble_vector(MPI_Comm comm, cl::sycl::queue& queue,
                        const memory::form_data_t& data, int verbose_mode = 1)
{

  std::string step{"Assemble vector on device"};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();
  experimental::sycl::la::AdjacencyList acc
      = experimental::sycl::la::compute_vector_acc_map(comm, queue, data);
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Create  accumulator from dofmap"] = (timer_end - timer_start);

  auto start = std::chrono::system_clock::now();

  timer_start = std::chrono::system_clock::now();

  // Allocated unassembled vector on device
  std::int32_t ndofs_ext = data.ndofs_cell * data.ncells;
  auto b_ext = cl::sycl::malloc_device<double>(ndofs_ext, queue);
  queue.fill<double>(b_ext, 0., ndofs_ext).wait();
  // Assemble local contributions
  assemble_vector_impl(queue, b_ext, data.x, data.xdofs, data.coeffs_L,
                       data.ncells, data.ndofs, data.ndofs_cell);
  timer_end = std::chrono::system_clock::now();
  timings["1 - Compute cell contributions"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  double* b = cl::sycl::malloc_shared<double>(data.ndofs, queue);
  accumulate_impl(queue, b, b_ext, acc.indptr, acc.indices, acc.num_nodes);
  timer_end = std::chrono::system_clock::now();
  timings["2 - Accumulate cells contributions"] = (timer_end - timer_start);

  // Free temporary device data
  cl::sycl::free(b_ext, queue);
  cl::sycl::free(acc.indices, queue);
  cl::sycl::free(acc.indptr, queue);

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);
  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);

  return b;
}

//--------------------------------------------------------------------------
// Submit vector assembly kernels to queue
void assemble_matrix(MPI_Comm comm, cl::sycl::queue& queue,
                     const memory::form_data_t& data,
                     experimental::sycl::la::CsrMatrix mat,
                     experimental::sycl::la::AdjacencyList acc_map,
                     int verbose_mode = 1)
{

  std::string step{"Assemble matrix on device"};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto start = std::chrono::system_clock::now();

  // Number of stored nonzeros on the extended COO format
  std::int32_t stored_nz = data.ncells * data.ndofs_cell * data.ndofs_cell;

  auto timer_start = std::chrono::system_clock::now();
  auto A_ext = cl::sycl::malloc_device<double>(stored_nz, queue);
  queue.fill<double>(A_ext, 0., stored_nz).wait_and_throw();
  assemble_matrix_impl(queue, A_ext, data.x, data.xdofs, data.coeffs_a,
                       data.ncells, data.ndofs, data.ndofs_cell);
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Compute cell contributions"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  accumulate_impl(queue, mat.data, A_ext, acc_map.indptr, acc_map.indices,
                  acc_map.num_nodes);
  timer_end = std::chrono::system_clock::now();
  timings["2 - Accumulate contributions"] = (timer_end - timer_start);

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);

  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);
}

//--------------------------------------------------------------------------
// Submit vector assembly kernels to queue
double* assemble_vector_atomic(MPI_Comm comm, cl::sycl::queue& queue,
                               const memory::form_data_t& data,
                               int verbose_mode = 1)
{

  std::string step{"Assemble vector on device with atomics."};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();

  double* b = cl::sycl::malloc_shared<double>(data.ndofs, queue);
  queue.fill<double>(b, 0., data.ndofs).wait();
  assemble_vector_search_impl(queue, b, data.x, data.xdofs, data.coeffs_L,
                              data.dofs, data.ncells, data.ndofs,
                              data.ndofs_cell);
  auto timer_end = std::chrono::system_clock::now();

  timings["Assemble cells and accumulate"] = (timer_end - timer_start);
  timings["Total"] = (timer_end - timer_start);
  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);
  return b;
}
//--------------------------------------------------------------------------
void assemble_matrix_search(MPI_Comm comm, cl::sycl::queue& queue,
                            const memory::form_data_t& data,
                            experimental::sycl::la::CsrMatrix mat,
                            int verbose_mode = 1)
{

  std::string step{"Assemble matrix on device with atomics."};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();
  assemble_matrix_search_impl(queue, mat.data, mat.indptr, mat.indices, data.x,
                              data.xdofs, data.coeffs_a, data.dofs, data.ncells,
                              data.ndofs, data.ndofs_cell);
  auto timer_end = std::chrono::system_clock::now();

  timings["Assemble cells and accumulate"] = (timer_end - timer_start);
  timings["Total"] = (timer_end - timer_start);

  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);
}
//--------------------------------------------------------------------------
void assemble_matrix_lookup(MPI_Comm comm, cl::sycl::queue& queue,
                            const memory::form_data_t& data,
                            experimental::sycl::la::CsrMatrix mat,
                            std::int32_t* lookup_table, int verbose_mode = 1)
{
  std::string step{"Assemble matrix on device with atomics."};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();

  assemble_matrix_lookup_impl(queue, mat.data, mat.indptr, mat.indices,
                              lookup_table, data.x, data.xdofs, data.coeffs_a,
                              data.dofs, data.ncells, data.ndofs,
                              data.ndofs_cell);

  auto timer_end = std::chrono::system_clock::now();
  timings["Total"] = (timer_end - timer_start);

  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);
}

} // namespace dolfinx::experimental::sycl::assemble