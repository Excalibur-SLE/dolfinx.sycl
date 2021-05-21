#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>

#include <iostream>
#include <math.h>
#include <numeric>
#include <string>

#include "../problem.h"

using namespace dolfinx;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  std::size_t nx = 20;
  if (argc >= 2)
    nx = std::stoi(argv[1]);

  auto mesh = std::make_shared<mesh::Mesh>(
      generation::BoxMesh::create(mpi_comm, {{{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}}}, {{nx, nx, nx}},
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

  fem::Function<PetscScalar> u(V);

  dolfinx::common::Timer t1("ZZZ Create PETSc matrix");
  Mat A = dolfinx::fem::create_matrix(a);
  MatZeroEntries(A);
  t1.stop();

  la::PETScVector b(*L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

  // Assemble Matrix
  fem::assemble_matrix(la::PETScMatrix::set_fn(A, INSERT_VALUES), a, {});
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  for (int i = 0; i < 1; i++) {
    dolfinx::common::Timer t2("ZZZ Assemble Matrix");
    fem::assemble_matrix(la::PETScMatrix::set_fn(A, INSERT_VALUES), a, {});
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    t2.stop();
  }

  std::vector<double> b_(num_dofs, 0);
  const std::vector<double> constants = pack_constants(L);
  const array2d<double> coeffs = pack_coefficients(L);
  // fem::assemble_vector<double>(b_, L, constants, coeffs);

  int id = L.integral_ids(dolfinx::fem::IntegralType::cell)[0];
  const auto& fn = L.kernel(dolfinx::fem::IntegralType::cell, id);
  const std::vector<std::int32_t>& active_cells = L.domains(dolfinx::fem::IntegralType::cell, id);
  std::shared_ptr<const fem::DofMap> dofmap = L.function_spaces().at(0)->dofmap();
  const graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const std::int32_t num_cells = mesh->topology().connectivity(3, 0)->num_nodes();

  const std::vector<std::uint32_t> cell_info(num_cells);

  dolfinx::common::Timer t3("ZZZ Assemble Vector");
  dolfinx::fem::impl::assemble_cells<double>(b_, mesh->geometry(), active_cells, dofs, 1, fn,
                                             constants, coeffs, cell_info);
  t3.stop();

  Vec v;
  VecDuplicate(b.vec(), &v);
  dolfinx::common::Timer t4("ZZZ Matrix Vector Multiplication");
  MatMult(A, f->vector(), v);
  t4.stop();

  // la::PETScKrylovSolver solver(mpi_comm);
  // la::PETScOptions::set("ksp_type", "gmres");
  // la::PETScOptions::set("pc_type", "jacobi");
  // la::PETScOptions::set("rtol", 1e-5);
  // solver.set_from_options();
  // solver.set_operator(A);
  // solver.solve(u.vector(), b.vec());

  // io::VTKFile file("u.pvd");
  // file.write(u);

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  MatDestroy(&A);

  double norm;

  VecNorm(v, NORM_1, &norm);

  if (mpi_rank == 0)
    std::cout << "Number of degrees of freedom: " << V->dofmap()->index_map->size_global()
              << std::endl;

  return 0;
}
