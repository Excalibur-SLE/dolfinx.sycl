// Copyright (C) 2020-2021 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#define HIPSYCL_EXT_ENABLE_ALL

#ifdef SYCL_DEVICE_ONLY
#undef SYCL_DEVICE_ONLY
#include <dolfinx.h>
#define SYCL_DEVICE_ONLY
#else
#include <dolfinx.h>
#endif

#include <iostream>
#include <math.h>
#include <numeric>

#include "dolfinx_sycl.hpp"
#include "problem.h"
#include "solve.hpp"

using namespace dolfinx;
using namespace dolfinx::experimental::sycl;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  std::size_t nx = 20;
  std::string platform = "cpu";
  if (argc == 2)
    nx = std::stoi(argv[1]);
  else if (argc == 3)
  {
    nx = std::stoi(argv[1]);
    platform = argv[2];
  }

  auto cmap = fem::create_coordinate_map(create_coordinate_map_problem);
  std::array<Eigen::Vector3d, 2> pts{Eigen::Vector3d(-1, -1, -1),
                                     Eigen::Vector3d(1.0, 1.0, 1.0)};

  auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
      mpi_comm, pts, {{nx, nx, nx}}, cmap, mesh::GhostMode::none));

  mesh->topology_mutable().create_entity_permutations();
  auto V = fem::create_functionspace(create_functionspace_form_problem_a, "u",
                                     mesh);

  auto f = std::make_shared<fem::Function<PetscScalar>>(V);
  f->interpolate([](auto& x) {
    return (12 * M_PI * M_PI + 1) * Eigen::cos(2 * M_PI * x.row(0))
           * Eigen::cos(2 * M_PI * x.row(1)) * Eigen::cos(2 * M_PI * x.row(2));
  });

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_problem_L, {V},
                                                  {{"f", f}, {}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_problem_a, {V, V},
                                                  {}, {}, {});

  auto queue = utils::select_queue(mpi_comm, platform);

  int verb_mode = 2;
  if (verb_mode)
  {
    utils::print_device_info(queue.get_device());
    utils::print_function_space_info(V);
  }

  // Send form data to device (Geometry, Dofmap, Coefficients)
  auto form_data = memory::send_form_data(mpi_comm, queue, *L, *a, verb_mode);

  // Send form data to device (Geometry, Dofmap, Coefficients)
  auto mat = dolfinx::experimental::sycl::la::create_csr_matrix(mpi_comm, queue,
                                                                form_data);

// Assemble vector on device
#ifdef USE_ATOMICS_LOOKUP
  std::int32_t* lookup = dolfinx::experimental::sycl::la::compute_lookup_table(
      queue, mat, form_data);
  double* b = assemble::assemble_vector_atomic(mpi_comm, queue, form_data);
  assemble::assemble_matrix_lookup(mpi_comm, queue, form_data, mat, lookup);
  cl::sycl::free(lookup, queue);
#elif USE_ATOMICS_SEARCH
  double* b = assemble::assemble_vector_atomic(mpi_comm, queue, form_data);
  assemble::assemble_matrix_search(mpi_comm, queue, form_data, mat);
#else
  auto acc_map = dolfinx::experimental::sycl::la::compute_matrix_acc_map(
      queue, mat, form_data);
  double* b = assemble::assemble_vector(mpi_comm, queue, form_data);
  assemble::assemble_matrix(mpi_comm, queue, form_data, mat, acc_map);
  cl::sycl::free(acc_map.indices, queue);
  cl::sycl::free(acc_map.indptr, queue);
#endif

  double* x = cl::sycl::malloc_device<double>(form_data.ndofs, queue);
  queue.submit(
      [&](cl::sycl::handler& h) { h.fill<double>(x, 0., form_data.ndofs); });
  queue.wait();

  auto device = queue.get_device();
  std::string executor = "omp";

  if (device.is_gpu())
    executor = "cuda";

  std::cout << "\nUsing " << executor << " executor.\n";

  double norm = solve::ginkgo(mat.data, mat.indptr, mat.indices, mat.nrows,
                              mat.nnz, b, x, executor);

  double ex_norm = 0;
  VecNorm(f->vector(), NORM_2, &ex_norm);

  std::cout << "\nNorm of the computed solution " << norm << "\n";
  std::cout << "Norm of the reference solution "
            << ex_norm / (12. * M_PI * M_PI + 1.) << "\n\n";

  // Free device data
  cl::sycl::free(b, queue);
  cl::sycl::free(mat.data, queue);
  cl::sycl::free(mat.indices, queue);
  cl::sycl::free(mat.indptr, queue);

  return 0;
}
