
#include "assemble_impl.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

// Need to include C file in same translation unit as lambda
#include "problem.c"

using atomic_ref
    = sycl::ONEAPI::atomic_ref<double, sycl::ONEAPI::memory_order::relaxed,
                               sycl::ONEAPI::memory_scope::system,
                               cl::sycl::access::address_space::global_space>;

namespace
{
int binary_search(int* arr, int left, int right, int x)
{
  while (left <= right)
  {
    int middle = left + (right - left) / 2;

    if (arr[middle] == x)
      return middle;

    if (arr[middle] < x)
      left = middle + 1;
    else
      right = middle - 1;
  }

  return -1;
}
} // namespace

//--------------------------------------------------------------------------
void assemble_vector_impl(cl::sycl::queue& queue, double* b, double* x,
                          int* x_coor, double* coeff, int ncells, int ndofs,
                          int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    constexpr int ndofs_cell = L_num_dofs;

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];
      double c[ndofs_cell] = {0};
      double be[ndofs_cell] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_coor[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      const int offset = i * nelem_dofs;
      tabulate_cell_L(be, &coeff[offset], c, cell_geom, nullptr, nullptr, 0);

      for (int j = 0; j < ndofs_cell; j++)
        b[offset + j] = be[j];
    };

    cgh.parallel_for<class AssemblyKernelUSM_b>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}
//--------------------------------------------------------------------------
// Second kernel to accumulate RHS for each dof
void accumulate_impl(cl::sycl::queue& queue, double* x, double* x_ext,
                     int* offsets, int* indices, int ndofs)
{
  cl::sycl::range<1> range{(std::size_t)ndofs};
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);

      double val = 0.0;
      for (int j = offsets[i]; j < offsets[i + 1]; ++j)
        val += x_ext[indices[j]];

      x[i] = val;
    };

    cgh.parallel_for<class AccumulationKernel_b>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}
//--------------------------------------------------------------------------
void assemble_matrix_impl(cl::sycl::queue& queue, double* A, double* x,
                          int* x_coor, double* coeff, int ncells, int ndofs,
                          int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    constexpr int ndofs_cell = a_num_dofs;

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];

      double c[ndofs_cell] = {0};
      double Ae[ndofs_cell * ndofs_cell] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_coor[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      const int pos_c = i * nelem_dofs;
      tabulate_cell_a(Ae, &coeff[pos_c], c, cell_geom, nullptr, nullptr, 0);

      const int pos_A = i * nelem_dofs * nelem_dofs;
      for (int j = 0; j < nelem_dofs * nelem_dofs; j++)
        A[pos_A + j] = Ae[j];
    };

    cgh.parallel_for<class AssemblyKernelUSM_A>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}
//--------------------------------------------------------------------------
void assemble_vector_search_impl(cl::sycl::queue& queue, double* b, double* x,
                                 int* x_coor, double* coeff, int* dofs,
                                 int ncells, int ndofs, int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    constexpr int ndofs_cell = L_num_dofs;

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];
      double c[ndofs_cell] = {0};
      double be[ndofs_cell] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_coor[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      const int offset = i * nelem_dofs;
      tabulate_cell_L(be, &coeff[offset], c, cell_geom, nullptr, nullptr, 0);

      for (int j = 0; j < ndofs_cell; j++)
      {
        int pos = dofs[offset + j];
        atomic_ref atomic_b(b[pos]);
        atomic_b += be[j];
      }
    };

    cgh.parallel_for<class AssemblyKernelUSMSEarch>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}
//--------------------------------------------------------------------------
void assemble_matrix_search_impl(cl::sycl::queue& queue, double* data,
                                 std::int32_t* indptr, std::int32_t* indices,
                                 double* x, int* x_dof, double* coeffs,
                                 int* dofs, int ncells, int ndofs,
                                 int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    constexpr int ndofs_cell = a_num_dofs;

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];

      double c[ndofs_cell] = {0};
      double Ae[ndofs_cell * ndofs_cell] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_dof[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      const int offset = i * nelem_dofs;
      tabulate_cell_a(Ae, &coeffs[offset], c, cell_geom, nullptr, nullptr, 0);

      for (int j = 0; j < nelem_dofs; j++)
      {
        int row = dofs[offset + j];
        int first = indptr[row];
        int last = indptr[row + 1];
        for (int k = 0; k < nelem_dofs; k++)
        {
          int ind = dofs[offset + k];
          int pos = binary_search(indices, first, last, ind);
          atomic_ref atomic_A(data[pos]);
          atomic_A += Ae[j * nelem_dofs + k];
        }
      }
    };

    cgh.parallel_for<class AssemblyKernelUSMSearch>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}