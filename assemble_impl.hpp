// Copyright (C) 2020 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT
#pragma once

#include <CL/sycl.hpp>

// Submit assembly kernels to queue
void assemble_vector_impl(cl::sycl::queue& queue, double* b, double* x,
                          int* x_dof, double* coeff, int ncells, int ndofs,
                          int nelem_dofs);

// Submit assembly kernels to queue
void assemble_matrix_impl(cl::sycl::queue& queue, double* A, double* x,
                          int* x_dof, double* coeff, int ncells, int ndofs,
                          int nelem_dofs);

// Submit accumulation kernels to queue
void accumulate_impl(cl::sycl::queue& queue, double* b, double* b_ext,
                     int* offset, int* indices, int ndofs);

// Submit accumulation kernels to queue
void assemble_vector_search_impl(cl::sycl::queue& queue, double* b, double* x,
                                 int* x_dof, double* coeff, int* dofs,
                                 int ncells, int ndofs, int nelem_dofs);

// Submit assembly kernels to queue
void assemble_matrix_search_impl(cl::sycl::queue& queue, double* data,
                                 std::int32_t* indptr, std::int32_t* indices,
                                 double* x, int* x_dof, double* coeffs,
                                 int* dofs, int ncells, int ndofs,
                                 int nelem_dofs);