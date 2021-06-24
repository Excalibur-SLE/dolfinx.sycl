// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>

#include "dolfinx_sycl.hpp"
#include "problem.h"

using namespace dolfinx;
using namespace dolfinx::experimental::sycl;

using insert_func_t = std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                                        const std::int32_t*, const double*)>;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    MPI_Comm mpi_comm{MPI_COMM_WORLD};

    int mpi_size, mpi_rank;
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    std::size_t nx = 20;
    std::string platform = "cpu";
    if (argc == 2)
      nx = std::stoi(argv[1]);
    else if (argc == 3) {
      nx = std::stoi(argv[1]);
      platform = argv[2];
    }

    auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        mpi_comm, {{{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}}}, {{nx, nx, nx}},
        mesh::CellType::tetrahedron, mesh::GhostMode::none));
    mesh->topology_mutable().create_entity_permutations();

    auto V = fem::create_functionspace(functionspace_form_problem_a, "u", mesh);

    std::size_t num_dofs
        = V->dofmap()->index_map->size_local() + V->dofmap()->index_map->num_ghosts();

    common::Timer t0("interpolate");
    auto f = std::make_shared<fem::Function<double>>(V);
    f->interpolate([](auto& x) {
      return (12 * M_PI * M_PI + 1) * xt::cos(2 * M_PI * xt::row(x, 0))
             * xt::cos(2 * M_PI * xt::row(x, 1)) * xt::cos(2 * M_PI * xt::row(x, 2));
    });
    t0.stop();

    auto k = std::make_shared<fem::Constant<PetscScalar>>(1.0);

    // Define variational forms
    auto L = dolfinx::fem::create_form<double>(*form_problem_L, {V}, {{"f", f}, {}}, {}, {});
    auto a = fem::create_form<double>(*form_problem_a, {V, V}, {}, {{"k", k}}, {});

    auto queue = utils::select_queue(mpi_comm, platform);
    utils::print_device_info(queue.get_device());
    utils::print_function_space_info(V);

    // Send form data to device (Geometry, Dofmap, Coefficients)
    auto form_data = memory::send_form_data(mpi_comm, queue, L, a);

    // Send form data to device (Geometry, Dofmap, Coefficients)
    auto mat = dolfinx::experimental::sycl::la::create_csr_matrix(mpi_comm, queue, form_data);

    // Assemble csr matrix using binary search and
    assemble::assemble_matrix(queue, form_data, mat);

    dolfinx::common::Timer t10("z Assemble Matrix SYCL");
    assemble::assemble_matrix(queue, form_data, mat);
    t10.stop();

    double* b = cl::sycl::malloc_device<double>(num_dofs, queue);
    dolfinx::common::Timer t11("z Assemble Vector SYCL");
    assemble::assemble_vector(queue, form_data, b);
    t11.stop();

    dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});
  }

  common::subsystem::finalize_mpi();
  return 0;
}
