// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>
#include <dolfinx.h>

#include "kernel.hpp"
#include "la.hpp"
#include "memory.hpp"

using namespace dolfinx::experimental::sycl;
using atomic_ref = sycl::ONEAPI::atomic_ref<double, sycl::ONEAPI::memory_order::relaxed,
                                            sycl::ONEAPI::memory_scope::system,
                                            cl::sycl::access::address_space::global_space>;

namespace {
int binary_search(int* arr, int left, int right, int x) {
  while (left <= right) {
    int middle = left + (right - left) / 2;

    if (arr[middle] == x)
      return middle;

    if (arr[middle] < x)
      left = middle + 1;
    else
      right = middle - 1;
  }

  return -1;
}
} // namespace

template <typename T, int dim>
using local_accessor
    = cl::sycl::accessor<T, dim, cl::sycl::access_mode::read_write, cl::sycl::target::local>;
namespace dolfinx::experimental::sycl::assemble {

void assemble_matrix(cl::sycl::queue& queue, const memory::form_data_t& data,
                     experimental::sycl::la::CsrMatrix mat) {
  dolfinx::common::Timer t0("x Assemble Matrix");
  std::size_t ncells = data.dofs.get_range()[0];
  std::size_t ndofs_cell = data.dofs.get_range()[1];
  constexpr std::int32_t gdim = 3;

  // get buffers
  cl::sycl::buffer<std::int32_t, 2> x_dofs = data.x_dofs;
  cl::sycl::buffer<std::int32_t, 2> dofs_ = data.dofs;
  cl::sycl::buffer<double, 2> x_ = data.x;

  cl::sycl::event event = queue.submit([&](cl::sycl::handler& h) {
    // accessors
    cl::sycl::accessor x{x_, h, cl::sycl::read_only};
    cl::sycl::accessor xdofs{x_dofs, h, cl::sycl::read_only};
    cl::sycl::accessor dofs{dofs_, h, cl::sycl::read_only};

    h.parallel_for(cl::sycl::range<1>(ncells), [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);

      double Ae[4][4] = {{0}};
      double cell_geom[12] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j) {
        const std::size_t pos = xdofs[i][j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[pos][k];
      }

      // Get local values
      tabulate_a(Ae, cell_geom);

      for (int j = 0; j < ndofs_cell; j++) {
        int row = dofs[i][j];
        for (int k = 0; k < ndofs_cell; k++) {
          int ind = dofs[i][k];
          int pos = binary_search(mat.indices, mat.indptr[row], mat.indptr[row + 1], ind);
          atomic_ref atomic_A(mat.data[pos]);
          atomic_A += Ae[j][k];
        }
      }
    });
  });

  try {
    queue.wait_and_throw();
  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
  }
  t0.stop();
}

} // namespace dolfinx::experimental::sycl::assemble