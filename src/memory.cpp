// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "memory.hpp"
#include <cassert>

using namespace dolfinx::experimental::sycl;

memory::form_data_t dolfinx::experimental::sycl::memory::send_form_data(
    MPI_Comm comm, cl::sycl::queue& queue, const fem::Form<double>& L,
    const fem::Form<double>& a, int verbose_mode)
{

  dolfinx::common::Timer t0("x Send data form data to device");
  assert(L.rank() == 1);
  assert(a.rank() == 2);

  auto mesh = L.mesh();
  auto dofmap = L.function_spaces()[0]->dofmap();
  int tdim = mesh->topology().dim();
  std::int32_t ndofs
      = dofmap->index_map->size_local() + dofmap->index_map->num_ghosts();
  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local()
                        + mesh->topology().index_map(tdim)->num_ghosts();
  int ndofs_cell = dofmap->list().num_links(0);

  // Send coordinates to device
  const auto& geometry = mesh->geometry().x();
  auto x_d = cl::sycl::malloc_device<double>(geometry.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_d, geometry.data(), sizeof(double) * geometry.size());
  });

  // Send geometry dofmap to device
  const auto& x_dofmap = mesh->geometry().dofmap().array();
  auto x_dofs_d = cl::sycl::malloc_device<std::int32_t>(x_dofmap.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_dofs_d, x_dofmap.data(), sizeof(std::int32_t) * x_dofmap.size());
  });

  // Send RHS coefficients to device
  auto coeffs = dolfinx::fem::pack_coefficients(L);
  auto coeff_d = cl::sycl::malloc_device<double>(coeffs.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_d, coeffs.data(), sizeof(double) * coeffs.size());
  });

  // Send LHS coefficients to device
  auto coeffs_a = dolfinx::fem::pack_coefficients(a);
  auto coeff_a_d = cl::sycl::malloc_device<double>(coeffs_a.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_a_d, coeffs_a.data(), sizeof(double) * coeffs_a.size());
  });

  // Send dofmap to device
  auto& dofs = dofmap->list().array();
  auto dofs_d = cl::sycl::malloc_device<std::int32_t>(dofs.size(), queue);
  auto e = queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(dofs_d, dofs.data(), sizeof(std::int32_t) * dofs.size());
  });

  queue.wait();
  t0.stop();

  return {x_d, x_dofs_d, coeff_d, coeff_a_d, dofs_d, ndofs, ncells, ndofs_cell};
}