// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <xtensor/xmath.hpp>

#include "dolfinx_sycl.hpp"
#include "problem.h"

using namespace dolfinx;
using namespace dolfinx::experimental::sycl;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  {
    MPI_Comm mpi_comm{MPI_COMM_WORLD};

    std::size_t nx = 2;

    auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        mpi_comm, {{{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}}}, {{nx, nx, nx}},
        mesh::CellType::tetrahedron, mesh::GhostMode::none));
    mesh->topology_mutable().create_entity_permutations();

    auto V = fem::create_functionspace(functionspace_form_problem_a, "u", mesh);

    std::size_t num_cells
        = mesh->topology().index_map(3)->size_local() + mesh->topology().index_map(3)->num_ghosts();
    std::size_t num_dofs
        = V->dofmap()->index_map->size_local() + V->dofmap()->index_map->num_ghosts();

    common::Timer t0("interpolate");
    std::shared_ptr<fem::Function<double>> f = std::make_shared<fem::Function<double>>(V);
    auto& m_array = f->x()->mutable_array();
    std::fill(m_array.begin(), m_array.end(), 3);
    auto k = std::make_shared<fem::Constant<PetscScalar>>(1.0);

    // Define variational forms
    auto L = dolfinx::fem::create_form<double>(*form_problem_L, {V}, {{"f", f}, {}}, {}, {});
    auto a = fem::create_form<double>(*form_problem_a, {V, V}, {}, {{"k", k}}, {});

    auto queue = cl::sycl::queue(cl::sycl::cpu_selector());

    // Send form data to device (Geometry, Dofmap, Coefficients)
    auto form_data = memory::send_form_data(mpi_comm, queue, L, a);

    // Send form data to device (Geometry, Dofmap, Coefficients)
    auto mat = dolfinx::experimental::sycl::la::create_csr_matrix(mpi_comm, queue, form_data);

    assert(form_data.ndofs == num_dofs);
    assert(mat.nrows == num_dofs);

    Mat A = dolfinx::fem::create_matrix(a);
    MatZeroEntries(A);
    fem::assemble_matrix(dolfinx::la::PETScMatrix::set_fn(A, ADD_VALUES), a, {});
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    MatInfo info;
    MatGetInfo(A, MAT_LOCAL, &info);

    assert(std::size_t(form_data.ndofs) == num_dofs);
    assert(std::size_t(form_data.ncells) == num_cells);
    assert(mat.nnz == info.nz_allocated);

    // Assemble csr matrix using binary search and
    assemble::assemble_matrix(queue, form_data, mat);

    double* array;
    MatSeqAIJGetArray(A, &array);

    double l1norm = 0;
    for (int i = 0; i < mat.nnz; i++)
      l1norm += std::fabs(array[i] - mat.data[i]);

    assert(l1norm < 1e-8);

    // Test Assemble Vector
    xt::xtensor<double, 1> b = xt::zeros<double>({num_dofs});
    dolfinx::fem::assemble_vector<double>(b, L);

    double* c = cl::sycl::malloc_device<double>(num_dofs, queue);
    queue.fill<double>(c, 0, num_dofs);
    assemble::assemble_vector(queue, form_data, c);
    queue.wait();

    auto _c = xt::adapt(c, num_dofs, xt::no_ownership(), std::vector<std::size_t>{num_dofs});

    assert(xt::allclose(b, _c));
  }

  common::subsystem::finalize_petsc();
  return 0;
}
