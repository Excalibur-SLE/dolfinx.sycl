// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "solve.hpp"
#include "timing.hpp"

#include <ginkgo/ginkgo.hpp>
#include <map>
#include <mpi.h>

double dolfinx::experimental::sycl::solve::ginkgo(
    double* A, std::int32_t* indptr, std::int32_t* indices, std::int32_t nrows,
    std::int32_t nnz, double* b, double* x, std::string executor)
{
  using mtx = gko::matrix::Csr<double, std::int32_t>;
  using cg = gko::solver::Cg<double>;

  std::string step{"Solve Using Ginkgo"};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto start = std::chrono::system_clock::now();

  std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
      exec_map{{"omp", [] { return gko::OmpExecutor::create(); }},
               {"cuda",
                [] {
                  return gko::CudaExecutor::create(
                      0, gko::OmpExecutor::create(), true);
                }},
               {"hip",
                [] {
                  return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
                }},
               {"dpcpp",
                [] {
                  return gko::DpcppExecutor::create(0,
                                                    gko::OmpExecutor::create());
                }},
               {"reference", [] { return gko::ReferenceExecutor::create(); }}};

  auto exec = exec_map.at(executor)(); // throws if not valid

  // Create Input Vector
  auto b_view = gko::Array<double>::view(exec, nrows, b);
  auto in = gko::matrix::Dense<double>::create(exec, gko::dim<2>(nrows, 1),
                                               b_view, 1);

  // Create Output Vector
  auto x_view = gko::Array<double>::view(exec, nrows, x);
  auto out = gko::matrix::Dense<double>::create(exec, gko::dim<2>(nrows, 1),
                                                x_view, 1);
  auto data_v = gko::Array<double>::view(exec, nnz, A);
  auto indptr_v = gko::Array<std::int32_t>::view(exec, nrows + 1, indptr);
  auto indices_v = gko::Array<std::int32_t>::view(exec, nnz, indices);
  auto dim = gko::dim<2>(nrows);
  auto matrix = mtx::create(exec, dim, data_v, indices_v, indptr_v);

  const double reduction_factor = 1e-5;

  auto solver_gen
      = cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(nrows).on(exec),
                gko::stop::ResidualNormReduction<double>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .on(exec);

  auto timer_start = std::chrono::system_clock::now();
  auto solver = solver_gen->generate(gko::give(matrix));
  solver->apply(gko::lend(in), gko::lend(out));
  auto timer_end = std::chrono::system_clock::now();

  timings["0 - Solve Linear System"] = (timer_end - timer_start);

  auto ref_exec = gko::ReferenceExecutor::create();
  auto res = gko::initialize<gko::matrix::Dense<double>>({0.0}, ref_exec);
  out->compute_norm2(gko::lend(res));

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);

  dolfinx::experimental::sycl::timing::print_timing_info(MPI_COMM_WORLD,
                                                         timings, step, 2);

  double norm = res->get_values()[0];
  
  return norm;
}