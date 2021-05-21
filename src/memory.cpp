// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "memory.hpp"
#include <cassert>

using namespace dolfinx::experimental::sycl;

memory::form_data_t dolfinx::experimental::sycl::memory::send_form_data(
    MPI_Comm comm, cl::sycl::queue& queue, const fem::Form<double>& L, const fem::Form<double>& a) {

  dolfinx::common::Timer t0("x Send data form data to device");
  assert(L.rank() == 1);
  assert(a.rank() == 2);

  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L.mesh();
  auto dofmap = L.function_spaces()[0]->dofmap();
  int tdim = mesh->topology().dim();
  std::size_t ndofs = dofmap->index_map->size_local() + dofmap->index_map->num_ghosts();
  std::size_t ncells = mesh->topology().index_map(tdim)->size_local()
                       + mesh->topology().index_map(tdim)->num_ghosts();
  std::size_t ndofs_cell = dofmap->list().num_links(0);

  // Send coordinates to device
  const xt::xtensor<double, 2>& geometry = mesh->geometry().x();
  cl::sycl::buffer<double, 2> x(geometry.data(), {geometry.shape(0), geometry.shape(1)});

  // Send geometry dofmap to device
  const auto& x_dofmap = mesh->geometry().dofmap().array();
  std::size_t nverts_cell = mesh->geometry().dofmap().num_links(0);
  cl::sycl::buffer<std::int32_t, 2> x_dofs(x_dofmap.data(), {ncells, nverts_cell});

  // Send RHS coefficients to device
  dolfinx::array2d<double> _coeffs = dolfinx::fem::pack_coefficients(L);
  cl::sycl::buffer<double, 2> coeffs({_coeffs.shape[0], _coeffs.shape[1]});
  cl::sycl::host_accessor coeff_acc(coeffs);
  for (int i = 0; i < coeff_acc.get_range()[0]; i++) {
    for (int j = 0; j < coeff_acc.get_range()[1]; j++) {
      coeff_acc[i][j] = _coeffs(i, j);
    }
  }
  // Send dofmap to device
  auto& _dofs = dofmap->list().array();
  cl::sycl::buffer<std::int32_t, 2> dofs(_dofs.data(), {ncells, ndofs_cell});

  return {x, x_dofs, coeffs, dofs, ndofs, ncells};
}