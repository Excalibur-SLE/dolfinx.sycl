#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>

#include <iostream>
#include <math.h>
#include <numeric>
#include <string>

#include "../problem.h"

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  std::size_t nx = 20;
  if (argc >= 2)
    nx = std::stoi(argv[1]);

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

  fem::Function<PetscScalar> u(V);

  Mat A = dolfinx::fem::create_matrix(*a);
  la::PETScVector b(*L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

  MatZeroEntries(A);

  // Assemble Matrix
  fem::assemble_matrix(la::PETScMatrix::add_fn(A), *a, {});
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  for (int i = 0; i < 5; i++)
  {
    dolfinx::common::Timer t0("ZZZ Assemble Matrix");
    fem::assemble_matrix(la::PETScMatrix::add_fn(A), *a, {});
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    t0.stop();
  }

  dolfinx::common::Timer t1("ZZZ Assemble Vector");
  fem::assemble_vector_petsc(b.vec(), *L);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  t1.stop();

  la::PETScKrylovSolver solver(mpi_comm);
  la::PETScOptions::set("ksp_type", "cg");
  la::PETScOptions::set("pc_type", "jacobi");
  la::PETScOptions::set("rtol", 1e-5);
  solver.set_from_options();
  solver.set_operator(A);
  solver.solve(u.vector(), b.vec());

  io::VTKFile file("u.pvd");
  file.write(u);

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  MatDestroy(&A);

  if (mpi_rank == 0)
    std::cout << "Number of degrees of freedom: "
              << V->dofmap()->index_map->size_global() << std::endl;

  return 0;
}
