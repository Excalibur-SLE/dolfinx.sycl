// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>
#include <dolfinx.h>

#include "kernel.hpp"
#include "la.hpp"
#include "memory.hpp"

using atomic_ref = sycl::ONEAPI::atomic_ref<double, sycl::ONEAPI::memory_order::relaxed,
                                            sycl::ONEAPI::memory_scope::system,
                                            cl::sycl::access::address_space::global_space>;

template <typename T, int dim>
using local_accessor
    = cl::sycl::accessor<T, dim, cl::sycl::access_mode::read_write, cl::sycl::target::local>;

using namespace dolfinx::experimental::sycl;

namespace dolfinx::experimental::sycl::assemble {
void assemble_vector(cl::sycl::queue& queue, const memory::form_data_t& data, double* b) {
  std::int32_t ncells = data.ncells;
  std::int32_t ndofs_cell = data.dofs.get_range()[1];
  constexpr std::int32_t gdim = 3;

  cl::sycl::buffer<std::int32_t, 2> x_dofs = data.x_dofs;
  cl::sycl::buffer<std::int32_t, 2> dofs_ = data.dofs;
  cl::sycl::buffer<double, 2> x_ = data.x;
  cl::sycl::buffer<double, 2> coeff_ = data.coeffs;

  cl::sycl::event e0 = queue.submit([&](cl::sycl::handler& h) {
    cl::sycl::accessor coeff{coeff_, h, cl::sycl::read_only};
    cl::sycl::accessor x{x_, h, cl::sycl::read_only};
    cl::sycl::accessor xdofs{x_dofs, h, cl::sycl::read_only};
    cl::sycl::accessor dofs{dofs_, h, cl::sycl::read_only};

    h.parallel_for(cl::sycl::range<1>(ncells), [=](cl::sycl::id<1> idx) {
      int i = idx.get(0);

      double cell_geom[12];
      double be[10] = {0};
      double w[10];

      for (std::size_t j = 0; j < ndofs_cell; ++j)
        w[j] = coeff[i][j];

      for (std::size_t j = 0; j < 4; ++j) {
        const std::size_t pos = xdofs[i][j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[pos][k];
      }

      // Get local values
      kernel(be, w, cell_geom);

      for (std::size_t j = 0; j < ndofs_cell; ++j) {
        int pos = dofs[i][j];
        atomic_ref atomic_b(b[pos]);
        atomic_b += be[j];
      }
    });
  });
}
} // namespace dolfinx::experimental::sycl::assemble