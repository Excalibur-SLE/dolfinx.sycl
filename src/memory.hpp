// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>
#include <cstdint>
#include <dolfinx.h>

using namespace dolfinx;

namespace dolfinx::experimental::sycl::memory {

struct form_data_t {
  cl::sycl::buffer<double, 2> x;
  cl::sycl::buffer<std::int32_t, 2> x_dofs;
  cl::sycl::buffer<double, 2> coeffs;
  cl::sycl::buffer<std::int32_t, 2> dofs;
  std::size_t ndofs;
  std::size_t ncells;
};

form_data_t send_form_data(MPI_Comm comm, cl::sycl::queue& queue, const fem::Form<double>& L,
                           const fem::Form<double>& a);

} // namespace dolfinx::experimental::sycl::memory
