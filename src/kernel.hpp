#pragma once

#define restrict __restrict__
using ufc_scalar_t = double;

void kernel(ufc_scalar_t* restrict A, const ufc_scalar_t* restrict w, const double* restrict coordinate_dofs)
{
    // Quadrature rules
    static const double weights_fad[4] = { 0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [permutation][entities][points][dofs]
    static const double FE8_C0_Qfad[1][1][4][4] =
        { { { { 0.1381966011250091, 0.5854101966249688, 0.138196601125011, 0.1381966011250109 },
              { 0.1381966011250091, 0.138196601125011, 0.5854101966249688, 0.1381966011250109 },
              { 0.1381966011250092, 0.1381966011250111, 0.1381966011250111, 0.585410196624969 },
              { 0.585410196624967, 0.1381966011250109, 0.138196601125011, 0.1381966011250109 } } } };
    static const double FE9_C0_D100_Qfad[1][1][1][4] = { { { { -1.0, 1.0, 0.0, 0.0 } } } };
    static const double FE9_C1_D010_Qfad[1][1][1][4] = { { { { -1.0, 0.0, 1.0, 0.0 } } } };
    static const double FE9_C2_D001_Qfad[1][1][1][4] = { { { { -1.0, 0.0, 0.0, 1.0 } } } };
    // Quadrature loop independent computations for quadrature rule fad
    const double J_c0 = coordinate_dofs[0] * FE9_C0_D100_Qfad[0][0][0][0] + coordinate_dofs[3] * FE9_C0_D100_Qfad[0][0][0][1] + coordinate_dofs[6] * FE9_C0_D100_Qfad[0][0][0][2] + coordinate_dofs[9] * FE9_C0_D100_Qfad[0][0][0][3];
    const double J_c4 = coordinate_dofs[1] * FE9_C1_D010_Qfad[0][0][0][0] + coordinate_dofs[4] * FE9_C1_D010_Qfad[0][0][0][1] + coordinate_dofs[7] * FE9_C1_D010_Qfad[0][0][0][2] + coordinate_dofs[10] * FE9_C1_D010_Qfad[0][0][0][3];
    const double J_c8 = coordinate_dofs[2] * FE9_C2_D001_Qfad[0][0][0][0] + coordinate_dofs[5] * FE9_C2_D001_Qfad[0][0][0][1] + coordinate_dofs[8] * FE9_C2_D001_Qfad[0][0][0][2] + coordinate_dofs[11] * FE9_C2_D001_Qfad[0][0][0][3];
    const double J_c5 = coordinate_dofs[1] * FE9_C2_D001_Qfad[0][0][0][0] + coordinate_dofs[4] * FE9_C2_D001_Qfad[0][0][0][1] + coordinate_dofs[7] * FE9_C2_D001_Qfad[0][0][0][2] + coordinate_dofs[10] * FE9_C2_D001_Qfad[0][0][0][3];
    const double J_c7 = coordinate_dofs[2] * FE9_C1_D010_Qfad[0][0][0][0] + coordinate_dofs[5] * FE9_C1_D010_Qfad[0][0][0][1] + coordinate_dofs[8] * FE9_C1_D010_Qfad[0][0][0][2] + coordinate_dofs[11] * FE9_C1_D010_Qfad[0][0][0][3];
    const double J_c1 = coordinate_dofs[0] * FE9_C1_D010_Qfad[0][0][0][0] + coordinate_dofs[3] * FE9_C1_D010_Qfad[0][0][0][1] + coordinate_dofs[6] * FE9_C1_D010_Qfad[0][0][0][2] + coordinate_dofs[9] * FE9_C1_D010_Qfad[0][0][0][3];
    const double J_c6 = coordinate_dofs[2] * FE9_C0_D100_Qfad[0][0][0][0] + coordinate_dofs[5] * FE9_C0_D100_Qfad[0][0][0][1] + coordinate_dofs[8] * FE9_C0_D100_Qfad[0][0][0][2] + coordinate_dofs[11] * FE9_C0_D100_Qfad[0][0][0][3];
    const double J_c3 = coordinate_dofs[1] * FE9_C0_D100_Qfad[0][0][0][0] + coordinate_dofs[4] * FE9_C0_D100_Qfad[0][0][0][1] + coordinate_dofs[7] * FE9_C0_D100_Qfad[0][0][0][2] + coordinate_dofs[10] * FE9_C0_D100_Qfad[0][0][0][3];
    const double J_c2 = coordinate_dofs[0] * FE9_C2_D001_Qfad[0][0][0][0] + coordinate_dofs[3] * FE9_C2_D001_Qfad[0][0][0][1] + coordinate_dofs[6] * FE9_C2_D001_Qfad[0][0][0][2] + coordinate_dofs[9] * FE9_C2_D001_Qfad[0][0][0][3];
    ufc_scalar_t sp_fad[15];
    sp_fad[0] = J_c4 * J_c8;
    sp_fad[1] = J_c5 * J_c7;
    sp_fad[2] = sp_fad[0] + -1 * sp_fad[1];
    sp_fad[3] = J_c0 * sp_fad[2];
    sp_fad[4] = J_c5 * J_c6;
    sp_fad[5] = J_c3 * J_c8;
    sp_fad[6] = sp_fad[4] + -1 * sp_fad[5];
    sp_fad[7] = J_c1 * sp_fad[6];
    sp_fad[8] = sp_fad[3] + sp_fad[7];
    sp_fad[9] = J_c3 * J_c7;
    sp_fad[10] = J_c4 * J_c6;
    sp_fad[11] = sp_fad[9] + -1 * sp_fad[10];
    sp_fad[12] = J_c2 * sp_fad[11];
    sp_fad[13] = sp_fad[8] + sp_fad[12];
    sp_fad[14] = fabs(sp_fad[13]);
    for (int iq = 0; iq < 4; ++iq)
    {
        // Quadrature loop body setup for quadrature rule fad
        // Varying computations for quadrature rule fad
        ufc_scalar_t w0 = 0.0;
        for (int ic = 0; ic < 4; ++ic)
            w0 += w[ic] * FE8_C0_Qfad[0][0][iq][ic];
        ufc_scalar_t sv_fad[1];
        sv_fad[0] = sp_fad[14] * w0;
        const ufc_scalar_t fw0 = sv_fad[0] * weights_fad[iq];
        for (int i = 0; i < 4; ++i)
            A[i] += fw0 * FE8_C0_Qfad[0][0][iq][i];
    }
}
template<typename M, typename N>
void tabulate_a(M A, const N coordinate_dofs)
{
    // Quadrature rules
    static const double weights_fad[4] = { 0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [permutation][entities][points][dofs]
    static const double FE8_C0_D100_Qfad[1][1][1][4] = { { { { -1.0, 1.0, 0.0, 0.0 } } } };
    static const double FE8_C0_Qfad[1][1][4][4] =
        { { { { 0.1381966011250091, 0.5854101966249688, 0.138196601125011, 0.1381966011250109 },
              { 0.1381966011250091, 0.138196601125011, 0.5854101966249688, 0.1381966011250109 },
              { 0.1381966011250092, 0.1381966011250111, 0.1381966011250111, 0.585410196624969 },
              { 0.585410196624967, 0.1381966011250109, 0.138196601125011, 0.1381966011250109 } } } };
    static const double FE9_C1_D010_Qfad[1][1][1][4] = { { { { -1.0, 0.0, 1.0, 0.0 } } } };
    static const double FE9_C2_D001_Qfad[1][1][1][4] = { { { { -1.0, 0.0, 0.0, 1.0 } } } };
    // Quadrature loop independent computations for quadrature rule fad
    const double J_c4 = coordinate_dofs[1] * FE9_C1_D010_Qfad[0][0][0][0] + coordinate_dofs[4] * FE9_C1_D010_Qfad[0][0][0][1] + coordinate_dofs[7] * FE9_C1_D010_Qfad[0][0][0][2] + coordinate_dofs[10] * FE9_C1_D010_Qfad[0][0][0][3];
    const double J_c8 = coordinate_dofs[2] * FE9_C2_D001_Qfad[0][0][0][0] + coordinate_dofs[5] * FE9_C2_D001_Qfad[0][0][0][1] + coordinate_dofs[8] * FE9_C2_D001_Qfad[0][0][0][2] + coordinate_dofs[11] * FE9_C2_D001_Qfad[0][0][0][3];
    const double J_c5 = coordinate_dofs[1] * FE9_C2_D001_Qfad[0][0][0][0] + coordinate_dofs[4] * FE9_C2_D001_Qfad[0][0][0][1] + coordinate_dofs[7] * FE9_C2_D001_Qfad[0][0][0][2] + coordinate_dofs[10] * FE9_C2_D001_Qfad[0][0][0][3];
    const double J_c7 = coordinate_dofs[2] * FE9_C1_D010_Qfad[0][0][0][0] + coordinate_dofs[5] * FE9_C1_D010_Qfad[0][0][0][1] + coordinate_dofs[8] * FE9_C1_D010_Qfad[0][0][0][2] + coordinate_dofs[11] * FE9_C1_D010_Qfad[0][0][0][3];
    const double J_c0 = coordinate_dofs[0] * FE8_C0_D100_Qfad[0][0][0][0] + coordinate_dofs[3] * FE8_C0_D100_Qfad[0][0][0][1] + coordinate_dofs[6] * FE8_C0_D100_Qfad[0][0][0][2] + coordinate_dofs[9] * FE8_C0_D100_Qfad[0][0][0][3];
    const double J_c1 = coordinate_dofs[0] * FE9_C1_D010_Qfad[0][0][0][0] + coordinate_dofs[3] * FE9_C1_D010_Qfad[0][0][0][1] + coordinate_dofs[6] * FE9_C1_D010_Qfad[0][0][0][2] + coordinate_dofs[9] * FE9_C1_D010_Qfad[0][0][0][3];
    const double J_c6 = coordinate_dofs[2] * FE8_C0_D100_Qfad[0][0][0][0] + coordinate_dofs[5] * FE8_C0_D100_Qfad[0][0][0][1] + coordinate_dofs[8] * FE8_C0_D100_Qfad[0][0][0][2] + coordinate_dofs[11] * FE8_C0_D100_Qfad[0][0][0][3];
    const double J_c3 = coordinate_dofs[1] * FE8_C0_D100_Qfad[0][0][0][0] + coordinate_dofs[4] * FE8_C0_D100_Qfad[0][0][0][1] + coordinate_dofs[7] * FE8_C0_D100_Qfad[0][0][0][2] + coordinate_dofs[10] * FE8_C0_D100_Qfad[0][0][0][3];
    const double J_c2 = coordinate_dofs[0] * FE9_C2_D001_Qfad[0][0][0][0] + coordinate_dofs[3] * FE9_C2_D001_Qfad[0][0][0][1] + coordinate_dofs[6] * FE9_C2_D001_Qfad[0][0][0][2] + coordinate_dofs[9] * FE9_C2_D001_Qfad[0][0][0][3];
    ufc_scalar_t sp_fad[80];
    sp_fad[0] = J_c4 * J_c8;
    sp_fad[1] = J_c5 * J_c7;
    sp_fad[2] = sp_fad[0] + -1 * sp_fad[1];
    sp_fad[3] = J_c0 * sp_fad[2];
    sp_fad[4] = J_c5 * J_c6;
    sp_fad[5] = J_c3 * J_c8;
    sp_fad[6] = sp_fad[4] + -1 * sp_fad[5];
    sp_fad[7] = J_c1 * sp_fad[6];
    sp_fad[8] = sp_fad[3] + sp_fad[7];
    sp_fad[9] = J_c3 * J_c7;
    sp_fad[10] = J_c4 * J_c6;
    sp_fad[11] = sp_fad[9] + -1 * sp_fad[10];
    sp_fad[12] = J_c2 * sp_fad[11];
    sp_fad[13] = sp_fad[8] + sp_fad[12];
    sp_fad[14] = sp_fad[2] / sp_fad[13];
    sp_fad[15] = J_c3 * (-1 * J_c8);
    sp_fad[16] = sp_fad[4] + sp_fad[15];
    sp_fad[17] = sp_fad[16] / sp_fad[13];
    sp_fad[18] = sp_fad[11] / sp_fad[13];
    sp_fad[19] = sp_fad[14] * sp_fad[14];
    sp_fad[20] = sp_fad[14] * sp_fad[17];
    sp_fad[21] = sp_fad[18] * sp_fad[14];
    sp_fad[22] = sp_fad[17] * sp_fad[17];
    sp_fad[23] = sp_fad[18] * sp_fad[17];
    sp_fad[24] = sp_fad[18] * sp_fad[18];
    sp_fad[25] = J_c2 * J_c7;
    sp_fad[26] = J_c8 * (-1 * J_c1);
    sp_fad[27] = sp_fad[25] + sp_fad[26];
    sp_fad[28] = sp_fad[27] / sp_fad[13];
    sp_fad[29] = J_c0 * J_c8;
    sp_fad[30] = J_c6 * (-1 * J_c2);
    sp_fad[31] = sp_fad[29] + sp_fad[30];
    sp_fad[32] = sp_fad[31] / sp_fad[13];
    sp_fad[33] = J_c1 * J_c6;
    sp_fad[34] = J_c0 * J_c7;
    sp_fad[35] = sp_fad[33] + -1 * sp_fad[34];
    sp_fad[36] = sp_fad[35] / sp_fad[13];
    sp_fad[37] = sp_fad[28] * sp_fad[28];
    sp_fad[38] = sp_fad[28] * sp_fad[32];
    sp_fad[39] = sp_fad[28] * sp_fad[36];
    sp_fad[40] = sp_fad[32] * sp_fad[32];
    sp_fad[41] = sp_fad[32] * sp_fad[36];
    sp_fad[42] = sp_fad[36] * sp_fad[36];
    sp_fad[43] = sp_fad[37] + sp_fad[19];
    sp_fad[44] = sp_fad[38] + sp_fad[20];
    sp_fad[45] = sp_fad[39] + sp_fad[21];
    sp_fad[46] = sp_fad[40] + sp_fad[22];
    sp_fad[47] = sp_fad[41] + sp_fad[23];
    sp_fad[48] = sp_fad[24] + sp_fad[42];
    sp_fad[49] = J_c1 * J_c5;
    sp_fad[50] = J_c2 * J_c4;
    sp_fad[51] = sp_fad[49] + -1 * sp_fad[50];
    sp_fad[52] = sp_fad[51] / sp_fad[13];
    sp_fad[53] = J_c2 * J_c3;
    sp_fad[54] = J_c0 * J_c5;
    sp_fad[55] = sp_fad[53] + -1 * sp_fad[54];
    sp_fad[56] = sp_fad[55] / sp_fad[13];
    sp_fad[57] = J_c0 * J_c4;
    sp_fad[58] = J_c1 * J_c3;
    sp_fad[59] = sp_fad[57] + -1 * sp_fad[58];
    sp_fad[60] = sp_fad[59] / sp_fad[13];
    sp_fad[61] = sp_fad[52] * sp_fad[52];
    sp_fad[62] = sp_fad[52] * sp_fad[56];
    sp_fad[63] = sp_fad[60] * sp_fad[52];
    sp_fad[64] = sp_fad[56] * sp_fad[56];
    sp_fad[65] = sp_fad[60] * sp_fad[56];
    sp_fad[66] = sp_fad[60] * sp_fad[60];
    sp_fad[67] = sp_fad[43] + sp_fad[61];
    sp_fad[68] = sp_fad[44] + sp_fad[62];
    sp_fad[69] = sp_fad[45] + sp_fad[63];
    sp_fad[70] = sp_fad[46] + sp_fad[64];
    sp_fad[71] = sp_fad[47] + sp_fad[65];
    sp_fad[72] = sp_fad[48] + sp_fad[66];
    sp_fad[73] = fabs(sp_fad[13]);
    sp_fad[74] = sp_fad[67] * sp_fad[73];
    sp_fad[75] = sp_fad[68] * sp_fad[73];
    sp_fad[76] = sp_fad[69] * sp_fad[73];
    sp_fad[77] = sp_fad[70] * sp_fad[73];
    sp_fad[78] = sp_fad[71] * sp_fad[73];
    sp_fad[79] = sp_fad[72] * sp_fad[73];
    for (int iq = 0; iq < 4; ++iq)
    {
        const ufc_scalar_t fw0 = sp_fad[73] * weights_fad[iq];
        const ufc_scalar_t fw1 = sp_fad[74] * weights_fad[iq];
        const ufc_scalar_t fw2 = sp_fad[75] * weights_fad[iq];
        const ufc_scalar_t fw3 = sp_fad[76] * weights_fad[iq];
        const ufc_scalar_t fw4 = sp_fad[77] * weights_fad[iq];
        const ufc_scalar_t fw5 = sp_fad[78] * weights_fad[iq];
        const ufc_scalar_t fw6 = sp_fad[79] * weights_fad[iq];
        ufc_scalar_t t0[4];
        ufc_scalar_t t1[4];
        ufc_scalar_t t2[4];
        ufc_scalar_t t3[4];
        for (int i = 0; i < 4; ++i)
        {
            t0[i] = fw0 * FE8_C0_Qfad[0][0][iq][i];
            t1[i] = fw1 * FE8_C0_D100_Qfad[0][0][0][i] + fw2 * FE9_C1_D010_Qfad[0][0][0][i] + fw3 * FE9_C2_D001_Qfad[0][0][0][i];
            t2[i] = fw2 * FE8_C0_D100_Qfad[0][0][0][i] + fw4 * FE9_C1_D010_Qfad[0][0][0][i] + fw5 * FE9_C2_D001_Qfad[0][0][0][i];
            t3[i] = fw3 * FE8_C0_D100_Qfad[0][0][0][i] + fw5 * FE9_C1_D010_Qfad[0][0][0][i] + fw6 * FE9_C2_D001_Qfad[0][0][0][i];
        }
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                ufc_scalar_t acc0 = 0;
                acc0 += FE8_C0_Qfad[0][0][iq][j] * t0[i];
                acc0 += FE8_C0_D100_Qfad[0][0][0][j] * t1[i];
                acc0 += FE9_C1_D010_Qfad[0][0][0][j] * t2[i];
                acc0 += FE9_C2_D001_Qfad[0][0][0][j] * t3[i];
                A[i][j] += acc0;
            }
        }
    }
}