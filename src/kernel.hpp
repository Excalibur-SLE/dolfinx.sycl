#pragma once

#define restrict __restrict__
using ufc_scalar_t = double;

void kernel(ufc_scalar_t* restrict A, const ufc_scalar_t* restrict w, const double* restrict coordinate_dofs)
{
    // Quadrature rules
    static const double weights_d8a[14] = { 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.01476497079049678, 0.01476497079049678, 0.01476497079049678, 0.01476497079049678, 0.02213979111426512, 0.02213979111426512, 0.02213979111426512, 0.02213979111426512 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [permutation][entities][points][dofs]
    static const double FE11_C0_Qd8a[1][1][14][10] =
        { { { { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
              { -0.08031550417191767, 0.2771604624527406, -0.08031550417191774, -0.0803155041719177, 0.04042252210657353, 0.2808394945810975, 0.2808394945810974, 0.04042252210657353, 0.04042252210657352, 0.2808394945810972 },
              { 0.2771604624527404, -0.08031550417191773, -0.08031550417191774, -0.08031550417191767, 0.04042252210657356, 0.04042252210657357, 0.04042252210657356, 0.2808394945810975, 0.2808394945810974, 0.2808394945810972 },
              { -0.08031550417191774, -0.08031550417191774, -0.08031550417191774, 0.2771604624527406, 0.2808394945810976, 0.2808394945810974, 0.0404225221065735, 0.2808394945810974, 0.0404225221065735, 0.04042252210657347 },
              { -0.08031550417191775, -0.08031550417191773, 0.2771604624527405, -0.08031550417191768, 0.2808394945810975, 0.04042252210657358, 0.2808394945810973, 0.04042252210657355, 0.2808394945810972, 0.0404225221065735 },
              { -0.116712266316459, -0.05041039684813046, -0.1167122663164589, -0.1167122663164589, 0.3953212143534666, 0.07152785091236924, 0.07152785091236921, 0.3953212143534666, 0.3953212143534665, 0.07152785091236913 },
              { -0.05041039684813049, -0.116712266316459, -0.1167122663164589, -0.1167122663164589, 0.3953212143534666, 0.3953212143534666, 0.3953212143534665, 0.07152785091236931, 0.07152785091236924, 0.07152785091236924 },
              { -0.116712266316459, -0.116712266316459, -0.116712266316459, -0.05041039684813045, 0.07152785091236921, 0.07152785091236924, 0.3953212143534665, 0.07152785091236924, 0.3953212143534665, 0.3953212143534665 },
              { -0.116712266316459, -0.116712266316459, -0.05041039684813052, -0.1167122663164589, 0.07152785091236936, 0.3953212143534666, 0.07152785091236927, 0.3953212143534666, 0.07152785091236927, 0.3953212143534665 } } } };
    static const double FE9_C0_D100_Qd8a[1][1][1][4] = { { { { -1.0, 1.0, 0.0, 0.0 } } } };
    static const double FE9_C1_D010_Qd8a[1][1][1][4] = { { { { -1.0, 0.0, 1.0, 0.0 } } } };
    static const double FE9_C2_D001_Qd8a[1][1][1][4] = { { { { -1.0, 0.0, 0.0, 1.0 } } } };
    // Quadrature loop independent computations for quadrature rule d8a
    const double J_c0 = coordinate_dofs[0] * FE9_C0_D100_Qd8a[0][0][0][0] + coordinate_dofs[3] * FE9_C0_D100_Qd8a[0][0][0][1] + coordinate_dofs[6] * FE9_C0_D100_Qd8a[0][0][0][2] + coordinate_dofs[9] * FE9_C0_D100_Qd8a[0][0][0][3];
    const double J_c4 = coordinate_dofs[1] * FE9_C1_D010_Qd8a[0][0][0][0] + coordinate_dofs[4] * FE9_C1_D010_Qd8a[0][0][0][1] + coordinate_dofs[7] * FE9_C1_D010_Qd8a[0][0][0][2] + coordinate_dofs[10] * FE9_C1_D010_Qd8a[0][0][0][3];
    const double J_c8 = coordinate_dofs[2] * FE9_C2_D001_Qd8a[0][0][0][0] + coordinate_dofs[5] * FE9_C2_D001_Qd8a[0][0][0][1] + coordinate_dofs[8] * FE9_C2_D001_Qd8a[0][0][0][2] + coordinate_dofs[11] * FE9_C2_D001_Qd8a[0][0][0][3];
    const double J_c5 = coordinate_dofs[1] * FE9_C2_D001_Qd8a[0][0][0][0] + coordinate_dofs[4] * FE9_C2_D001_Qd8a[0][0][0][1] + coordinate_dofs[7] * FE9_C2_D001_Qd8a[0][0][0][2] + coordinate_dofs[10] * FE9_C2_D001_Qd8a[0][0][0][3];
    const double J_c7 = coordinate_dofs[2] * FE9_C1_D010_Qd8a[0][0][0][0] + coordinate_dofs[5] * FE9_C1_D010_Qd8a[0][0][0][1] + coordinate_dofs[8] * FE9_C1_D010_Qd8a[0][0][0][2] + coordinate_dofs[11] * FE9_C1_D010_Qd8a[0][0][0][3];
    const double J_c1 = coordinate_dofs[0] * FE9_C1_D010_Qd8a[0][0][0][0] + coordinate_dofs[3] * FE9_C1_D010_Qd8a[0][0][0][1] + coordinate_dofs[6] * FE9_C1_D010_Qd8a[0][0][0][2] + coordinate_dofs[9] * FE9_C1_D010_Qd8a[0][0][0][3];
    const double J_c6 = coordinate_dofs[2] * FE9_C0_D100_Qd8a[0][0][0][0] + coordinate_dofs[5] * FE9_C0_D100_Qd8a[0][0][0][1] + coordinate_dofs[8] * FE9_C0_D100_Qd8a[0][0][0][2] + coordinate_dofs[11] * FE9_C0_D100_Qd8a[0][0][0][3];
    const double J_c3 = coordinate_dofs[1] * FE9_C0_D100_Qd8a[0][0][0][0] + coordinate_dofs[4] * FE9_C0_D100_Qd8a[0][0][0][1] + coordinate_dofs[7] * FE9_C0_D100_Qd8a[0][0][0][2] + coordinate_dofs[10] * FE9_C0_D100_Qd8a[0][0][0][3];
    const double J_c2 = coordinate_dofs[0] * FE9_C2_D001_Qd8a[0][0][0][0] + coordinate_dofs[3] * FE9_C2_D001_Qd8a[0][0][0][1] + coordinate_dofs[6] * FE9_C2_D001_Qd8a[0][0][0][2] + coordinate_dofs[9] * FE9_C2_D001_Qd8a[0][0][0][3];
    ufc_scalar_t sp_d8a[15];
    sp_d8a[0] = J_c4 * J_c8;
    sp_d8a[1] = J_c5 * J_c7;
    sp_d8a[2] = sp_d8a[0] + -1 * sp_d8a[1];
    sp_d8a[3] = J_c0 * sp_d8a[2];
    sp_d8a[4] = J_c5 * J_c6;
    sp_d8a[5] = J_c3 * J_c8;
    sp_d8a[6] = sp_d8a[4] + -1 * sp_d8a[5];
    sp_d8a[7] = J_c1 * sp_d8a[6];
    sp_d8a[8] = sp_d8a[3] + sp_d8a[7];
    sp_d8a[9] = J_c3 * J_c7;
    sp_d8a[10] = J_c4 * J_c6;
    sp_d8a[11] = sp_d8a[9] + -1 * sp_d8a[10];
    sp_d8a[12] = J_c2 * sp_d8a[11];
    sp_d8a[13] = sp_d8a[8] + sp_d8a[12];
    sp_d8a[14] = fabs(sp_d8a[13]);
    for (int iq = 0; iq < 14; ++iq)
    {
        // Quadrature loop body setup for quadrature rule d8a
        // Varying computations for quadrature rule d8a
        ufc_scalar_t w0 = 0.0;
        for (int ic = 0; ic < 10; ++ic)
            w0 += w[ic] * FE11_C0_Qd8a[0][0][iq][ic];
        ufc_scalar_t sv_d8a[1];
        sv_d8a[0] = sp_d8a[14] * w0;
        const ufc_scalar_t fw0 = sv_d8a[0] * weights_d8a[iq];
        for (int i = 0; i < 10; ++i)
            A[i] += fw0 * FE11_C0_Qd8a[0][0][iq][i];
    }
}
template<typename M, typename N>
void tabulate_a(M A, const N coordinate_dofs)
{
    // Quadrature rules
    static const double weights_d8a[14] = { 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.01476497079049678, 0.01476497079049678, 0.01476497079049678, 0.01476497079049678, 0.02213979111426512, 0.02213979111426512, 0.02213979111426512, 0.02213979111426512 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [permutation][entities][points][dofs]
    static const double FE17_C0_D001_Qd8a[1][1][14][10] =
        { { { { 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, -2.0, -2.0, 0.0 },
              { 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, -2.0, 0.0, -2.0 },
              { 1.0, 0.0, 0.0, -1.0, 2.0, 2.0, 0.0, 0.0, -2.0, -1.999999999999999 },
              { -1.0, 0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 2.0, 0.0, -2.0 },
              { -1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0 },
              { -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
              { 0.5978929390991826, 0.0, 0.0, -0.5978929390991822, 0.4021070609008178, 2.793678817297546, 0.0, 0.0, -0.402107060900818, -2.793678817297546 },
              { -1.793678817297546, 0.0, 0.0, -0.5978929390991821, 0.4021070609008178, 0.4021070609008177, 0.0, 2.391571756396728, -0.4021070609008181, -0.4021070609008178 },
              { 0.5978929390991823, 0.0, 0.0, 1.793678817297546, 0.402107060900818, 0.4021070609008179, 0.0, -2.391571756396728, -0.4021070609008179, -0.4021070609008178 },
              { 0.597892939099182, 0.0, 0.0, -0.5978929390991823, 2.793678817297546, 0.4021070609008187, 0.0, 0.0, -2.793678817297546, -0.4021070609008173 },
              { -0.2574914939727688, 0.0, 0.0, 0.2574914939727688, 1.257491493972769, 0.2275255180816937, 0.0, 0.0, -1.257491493972769, -0.2275255180816933 },
              { 0.7724744819183066, 0.0, 0.0, 0.2574914939727688, 1.257491493972769, 1.257491493972769, 0.0, -1.029965975891075, -1.257491493972768, -1.257491493972768 },
              { -0.2574914939727687, 0.0, 0.0, -0.7724744819183067, 1.257491493972769, 1.257491493972769, 0.0, 1.029965975891076, -1.257491493972769, -1.257491493972768 },
              { -0.2574914939727686, 0.0, 0.0, 0.2574914939727688, 0.2275255180816938, 1.257491493972769, 0.0, 0.0, -0.2275255180816936, -1.257491493972769 } } } };
    static const double FE17_C0_D010_Qd8a[1][1][14][10] =
        { { { { 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, -2.0, -2.0, 0.0 },
              { 1.0, 0.0, -1.0, 0.0, 2.0, 0.0, 2.0, -2.0, 0.0, -2.0 },
              { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, -1.999999999999999 },
              { -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.999999999999999, 0.0, 2.0, -2.0 },
              { -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
              { -1.0, 0.0, -1.0, 0.0, 1.999999999999999, 0.0, 0.0, -2.0, 2.0, 0.0 },
              { 0.5978929390991826, 0.0, -0.5978929390991822, 0.0, 0.402107060900818, 0.0, 2.793678817297546, -0.4021070609008177, 0.0, -2.793678817297546 },
              { -1.793678817297546, 0.0, -0.597892939099182, 0.0, 0.4021070609008174, 0.0, 0.4021070609008172, -0.4021070609008186, 2.391571756396728, -0.4021070609008183 },
              { 0.5978929390991823, 0.0, -0.597892939099182, 0.0, 2.793678817297546, 0.0, 0.4021070609008178, -2.793678817297546, 0.0, -0.4021070609008177 },
              { 0.597892939099182, 0.0, 1.793678817297546, 0.0, 0.4021070609008187, 0.0, 0.4021070609008182, -0.4021070609008175, -2.391571756396728, -0.4021070609008172 },
              { -0.2574914939727687, 0.0, 0.2574914939727688, 0.0, 1.257491493972769, 0.0, 0.2275255180816936, -1.257491493972769, 0.0, -0.2275255180816935 },
              { 0.7724744819183066, 0.0, 0.2574914939727687, 0.0, 1.257491493972769, 0.0, 1.257491493972769, -1.257491493972768, -1.029965975891075, -1.257491493972768 },
              { -0.2574914939727687, 0.0, 0.2574914939727687, 0.0, 0.2275255180816935, 0.0, 1.257491493972769, -0.2275255180816935, 0.0, -1.257491493972768 },
              { -0.2574914939727685, 0.0, -0.7724744819183064, 0.0, 1.257491493972768, 0.0, 1.257491493972768, -1.257491493972769, 1.029965975891075, -1.257491493972769 } } } };
    static const double FE17_C0_D100_Qd8a[1][1][14][10] =
        { { { { 1.0, -1.0, 0.0, 0.0, 0.0, 2.0, 2.0, -2.0, -2.0, 0.0 },
              { 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0, -2.0 },
              { 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, -2.0 },
              { -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
              { -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 2.0 },
              { -1.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0, 2.0 },
              { 0.5978929390991826, 1.793678817297547, 0.0, 0.0, 0.0, 0.4021070609008178, 0.4021070609008179, -0.4021070609008178, -0.4021070609008179, -2.391571756396729 },
              { -1.793678817297546, -0.5978929390991824, 0.0, 0.0, 0.0, 0.402107060900818, 0.4021070609008177, -0.4021070609008183, -0.4021070609008179, 2.391571756396728 },
              { 0.5978929390991825, -0.5978929390991824, 0.0, 0.0, 0.0, 2.793678817297546, 0.4021070609008179, -2.793678817297546, -0.4021070609008179, 0.0 },
              { 0.597892939099182, -0.5978929390991824, 0.0, 0.0, 0.0, 0.4021070609008185, 2.793678817297546, -0.4021070609008174, -2.793678817297546, 0.0 },
              { -0.2574914939727687, -0.7724744819183068, 0.0, 0.0, 0.0, 1.257491493972769, 1.257491493972769, -1.257491493972769, -1.257491493972769, 1.029965975891076 },
              { 0.7724744819183068, 0.2574914939727687, 0.0, 0.0, 0.0, 1.257491493972769, 1.257491493972769, -1.257491493972768, -1.257491493972769, -1.029965975891075 },
              { -0.2574914939727687, 0.2574914939727687, 0.0, 0.0, 0.0, 0.2275255180816938, 1.257491493972769, -0.2275255180816935, -1.257491493972769, 0.0 },
              { -0.2574914939727686, 0.2574914939727687, 0.0, 0.0, 0.0, 1.257491493972769, 0.2275255180816935, -1.257491493972769, -0.2275255180816936, 0.0 } } } };
    static const double FE17_C0_Qd8a[1][1][14][10] =
        { { { { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
              { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
              { -0.08031550417191767, 0.2771604624527406, -0.08031550417191774, -0.0803155041719177, 0.04042252210657353, 0.2808394945810975, 0.2808394945810974, 0.04042252210657353, 0.04042252210657352, 0.2808394945810972 },
              { 0.2771604624527404, -0.08031550417191773, -0.08031550417191774, -0.08031550417191767, 0.04042252210657356, 0.04042252210657357, 0.04042252210657356, 0.2808394945810975, 0.2808394945810974, 0.2808394945810972 },
              { -0.08031550417191774, -0.08031550417191774, -0.08031550417191774, 0.2771604624527406, 0.2808394945810976, 0.2808394945810974, 0.0404225221065735, 0.2808394945810974, 0.0404225221065735, 0.04042252210657347 },
              { -0.08031550417191775, -0.08031550417191773, 0.2771604624527405, -0.08031550417191768, 0.2808394945810975, 0.04042252210657358, 0.2808394945810973, 0.04042252210657355, 0.2808394945810972, 0.0404225221065735 },
              { -0.116712266316459, -0.05041039684813046, -0.1167122663164589, -0.1167122663164589, 0.3953212143534666, 0.07152785091236924, 0.07152785091236921, 0.3953212143534666, 0.3953212143534665, 0.07152785091236913 },
              { -0.05041039684813049, -0.116712266316459, -0.1167122663164589, -0.1167122663164589, 0.3953212143534666, 0.3953212143534666, 0.3953212143534665, 0.07152785091236931, 0.07152785091236924, 0.07152785091236924 },
              { -0.116712266316459, -0.116712266316459, -0.116712266316459, -0.05041039684813045, 0.07152785091236921, 0.07152785091236924, 0.3953212143534665, 0.07152785091236924, 0.3953212143534665, 0.3953212143534665 },
              { -0.116712266316459, -0.116712266316459, -0.05041039684813052, -0.1167122663164589, 0.07152785091236936, 0.3953212143534666, 0.07152785091236927, 0.3953212143534666, 0.07152785091236927, 0.3953212143534665 } } } };
    static const double FE9_C0_D100_Qd8a[1][1][1][4] = { { { { -1.0, 1.0, 0.0, 0.0 } } } };
    static const double FE9_C1_D010_Qd8a[1][1][1][4] = { { { { -1.0, 0.0, 1.0, 0.0 } } } };
    static const double FE9_C2_D001_Qd8a[1][1][1][4] = { { { { -1.0, 0.0, 0.0, 1.0 } } } };
    // Quadrature loop independent computations for quadrature rule d8a
    const double J_c4 = coordinate_dofs[1] * FE9_C1_D010_Qd8a[0][0][0][0] + coordinate_dofs[4] * FE9_C1_D010_Qd8a[0][0][0][1] + coordinate_dofs[7] * FE9_C1_D010_Qd8a[0][0][0][2] + coordinate_dofs[10] * FE9_C1_D010_Qd8a[0][0][0][3];
    const double J_c8 = coordinate_dofs[2] * FE9_C2_D001_Qd8a[0][0][0][0] + coordinate_dofs[5] * FE9_C2_D001_Qd8a[0][0][0][1] + coordinate_dofs[8] * FE9_C2_D001_Qd8a[0][0][0][2] + coordinate_dofs[11] * FE9_C2_D001_Qd8a[0][0][0][3];
    const double J_c5 = coordinate_dofs[1] * FE9_C2_D001_Qd8a[0][0][0][0] + coordinate_dofs[4] * FE9_C2_D001_Qd8a[0][0][0][1] + coordinate_dofs[7] * FE9_C2_D001_Qd8a[0][0][0][2] + coordinate_dofs[10] * FE9_C2_D001_Qd8a[0][0][0][3];
    const double J_c7 = coordinate_dofs[2] * FE9_C1_D010_Qd8a[0][0][0][0] + coordinate_dofs[5] * FE9_C1_D010_Qd8a[0][0][0][1] + coordinate_dofs[8] * FE9_C1_D010_Qd8a[0][0][0][2] + coordinate_dofs[11] * FE9_C1_D010_Qd8a[0][0][0][3];
    const double J_c0 = coordinate_dofs[0] * FE9_C0_D100_Qd8a[0][0][0][0] + coordinate_dofs[3] * FE9_C0_D100_Qd8a[0][0][0][1] + coordinate_dofs[6] * FE9_C0_D100_Qd8a[0][0][0][2] + coordinate_dofs[9] * FE9_C0_D100_Qd8a[0][0][0][3];
    const double J_c1 = coordinate_dofs[0] * FE9_C1_D010_Qd8a[0][0][0][0] + coordinate_dofs[3] * FE9_C1_D010_Qd8a[0][0][0][1] + coordinate_dofs[6] * FE9_C1_D010_Qd8a[0][0][0][2] + coordinate_dofs[9] * FE9_C1_D010_Qd8a[0][0][0][3];
    const double J_c6 = coordinate_dofs[2] * FE9_C0_D100_Qd8a[0][0][0][0] + coordinate_dofs[5] * FE9_C0_D100_Qd8a[0][0][0][1] + coordinate_dofs[8] * FE9_C0_D100_Qd8a[0][0][0][2] + coordinate_dofs[11] * FE9_C0_D100_Qd8a[0][0][0][3];
    const double J_c3 = coordinate_dofs[1] * FE9_C0_D100_Qd8a[0][0][0][0] + coordinate_dofs[4] * FE9_C0_D100_Qd8a[0][0][0][1] + coordinate_dofs[7] * FE9_C0_D100_Qd8a[0][0][0][2] + coordinate_dofs[10] * FE9_C0_D100_Qd8a[0][0][0][3];
    const double J_c2 = coordinate_dofs[0] * FE9_C2_D001_Qd8a[0][0][0][0] + coordinate_dofs[3] * FE9_C2_D001_Qd8a[0][0][0][1] + coordinate_dofs[6] * FE9_C2_D001_Qd8a[0][0][0][2] + coordinate_dofs[9] * FE9_C2_D001_Qd8a[0][0][0][3];
    ufc_scalar_t sp_d8a[80];
    sp_d8a[0] = J_c4 * J_c8;
    sp_d8a[1] = J_c5 * J_c7;
    sp_d8a[2] = sp_d8a[0] + -1 * sp_d8a[1];
    sp_d8a[3] = J_c0 * sp_d8a[2];
    sp_d8a[4] = J_c5 * J_c6;
    sp_d8a[5] = J_c3 * J_c8;
    sp_d8a[6] = sp_d8a[4] + -1 * sp_d8a[5];
    sp_d8a[7] = J_c1 * sp_d8a[6];
    sp_d8a[8] = sp_d8a[3] + sp_d8a[7];
    sp_d8a[9] = J_c3 * J_c7;
    sp_d8a[10] = J_c4 * J_c6;
    sp_d8a[11] = sp_d8a[9] + -1 * sp_d8a[10];
    sp_d8a[12] = J_c2 * sp_d8a[11];
    sp_d8a[13] = sp_d8a[8] + sp_d8a[12];
    sp_d8a[14] = sp_d8a[2] / sp_d8a[13];
    sp_d8a[15] = J_c3 * (-1 * J_c8);
    sp_d8a[16] = sp_d8a[4] + sp_d8a[15];
    sp_d8a[17] = sp_d8a[16] / sp_d8a[13];
    sp_d8a[18] = sp_d8a[11] / sp_d8a[13];
    sp_d8a[19] = sp_d8a[14] * sp_d8a[14];
    sp_d8a[20] = sp_d8a[14] * sp_d8a[17];
    sp_d8a[21] = sp_d8a[18] * sp_d8a[14];
    sp_d8a[22] = sp_d8a[17] * sp_d8a[17];
    sp_d8a[23] = sp_d8a[18] * sp_d8a[17];
    sp_d8a[24] = sp_d8a[18] * sp_d8a[18];
    sp_d8a[25] = J_c2 * J_c7;
    sp_d8a[26] = J_c8 * (-1 * J_c1);
    sp_d8a[27] = sp_d8a[25] + sp_d8a[26];
    sp_d8a[28] = sp_d8a[27] / sp_d8a[13];
    sp_d8a[29] = J_c0 * J_c8;
    sp_d8a[30] = J_c6 * (-1 * J_c2);
    sp_d8a[31] = sp_d8a[29] + sp_d8a[30];
    sp_d8a[32] = sp_d8a[31] / sp_d8a[13];
    sp_d8a[33] = J_c1 * J_c6;
    sp_d8a[34] = J_c0 * J_c7;
    sp_d8a[35] = sp_d8a[33] + -1 * sp_d8a[34];
    sp_d8a[36] = sp_d8a[35] / sp_d8a[13];
    sp_d8a[37] = sp_d8a[28] * sp_d8a[28];
    sp_d8a[38] = sp_d8a[28] * sp_d8a[32];
    sp_d8a[39] = sp_d8a[28] * sp_d8a[36];
    sp_d8a[40] = sp_d8a[32] * sp_d8a[32];
    sp_d8a[41] = sp_d8a[32] * sp_d8a[36];
    sp_d8a[42] = sp_d8a[36] * sp_d8a[36];
    sp_d8a[43] = sp_d8a[37] + sp_d8a[19];
    sp_d8a[44] = sp_d8a[38] + sp_d8a[20];
    sp_d8a[45] = sp_d8a[39] + sp_d8a[21];
    sp_d8a[46] = sp_d8a[40] + sp_d8a[22];
    sp_d8a[47] = sp_d8a[41] + sp_d8a[23];
    sp_d8a[48] = sp_d8a[24] + sp_d8a[42];
    sp_d8a[49] = J_c1 * J_c5;
    sp_d8a[50] = J_c2 * J_c4;
    sp_d8a[51] = sp_d8a[49] + -1 * sp_d8a[50];
    sp_d8a[52] = sp_d8a[51] / sp_d8a[13];
    sp_d8a[53] = J_c2 * J_c3;
    sp_d8a[54] = J_c0 * J_c5;
    sp_d8a[55] = sp_d8a[53] + -1 * sp_d8a[54];
    sp_d8a[56] = sp_d8a[55] / sp_d8a[13];
    sp_d8a[57] = J_c0 * J_c4;
    sp_d8a[58] = J_c1 * J_c3;
    sp_d8a[59] = sp_d8a[57] + -1 * sp_d8a[58];
    sp_d8a[60] = sp_d8a[59] / sp_d8a[13];
    sp_d8a[61] = sp_d8a[52] * sp_d8a[52];
    sp_d8a[62] = sp_d8a[52] * sp_d8a[56];
    sp_d8a[63] = sp_d8a[60] * sp_d8a[52];
    sp_d8a[64] = sp_d8a[56] * sp_d8a[56];
    sp_d8a[65] = sp_d8a[60] * sp_d8a[56];
    sp_d8a[66] = sp_d8a[60] * sp_d8a[60];
    sp_d8a[67] = sp_d8a[43] + sp_d8a[61];
    sp_d8a[68] = sp_d8a[44] + sp_d8a[62];
    sp_d8a[69] = sp_d8a[45] + sp_d8a[63];
    sp_d8a[70] = sp_d8a[46] + sp_d8a[64];
    sp_d8a[71] = sp_d8a[47] + sp_d8a[65];
    sp_d8a[72] = sp_d8a[48] + sp_d8a[66];
    sp_d8a[73] = fabs(sp_d8a[13]);
    sp_d8a[74] = sp_d8a[67] * sp_d8a[73];
    sp_d8a[75] = sp_d8a[68] * sp_d8a[73];
    sp_d8a[76] = sp_d8a[69] * sp_d8a[73];
    sp_d8a[77] = sp_d8a[70] * sp_d8a[73];
    sp_d8a[78] = sp_d8a[71] * sp_d8a[73];
    sp_d8a[79] = sp_d8a[72] * sp_d8a[73];
    for (int iq = 0; iq < 14; ++iq)
    {
        const ufc_scalar_t fw0 = sp_d8a[73] * weights_d8a[iq];
        const ufc_scalar_t fw1 = sp_d8a[74] * weights_d8a[iq];
        const ufc_scalar_t fw2 = sp_d8a[75] * weights_d8a[iq];
        const ufc_scalar_t fw3 = sp_d8a[76] * weights_d8a[iq];
        const ufc_scalar_t fw4 = sp_d8a[77] * weights_d8a[iq];
        const ufc_scalar_t fw5 = sp_d8a[78] * weights_d8a[iq];
        const ufc_scalar_t fw6 = sp_d8a[79] * weights_d8a[iq];
        ufc_scalar_t t0[10];
        ufc_scalar_t t1[10];
        ufc_scalar_t t2[10];
        ufc_scalar_t t3[10];
        for (int i = 0; i < 10; ++i)
        {
            t0[i] = fw0 * FE17_C0_Qd8a[0][0][iq][i];
            t1[i] = fw1 * FE17_C0_D100_Qd8a[0][0][iq][i] + fw2 * FE17_C0_D010_Qd8a[0][0][iq][i] + fw3 * FE17_C0_D001_Qd8a[0][0][iq][i];
            t2[i] = fw2 * FE17_C0_D100_Qd8a[0][0][iq][i] + fw4 * FE17_C0_D010_Qd8a[0][0][iq][i] + fw5 * FE17_C0_D001_Qd8a[0][0][iq][i];
            t3[i] = fw3 * FE17_C0_D100_Qd8a[0][0][iq][i] + fw5 * FE17_C0_D010_Qd8a[0][0][iq][i] + fw6 * FE17_C0_D001_Qd8a[0][0][iq][i];
        }
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                ufc_scalar_t acc0 = 0;
                acc0 += FE17_C0_Qd8a[0][0][iq][j] * t0[i];
                acc0 += FE17_C0_D100_Qd8a[0][0][iq][j] * t1[i];
                acc0 += FE17_C0_D010_Qd8a[0][0][iq][j] * t2[i];
                acc0 += FE17_C0_D001_Qd8a[0][0][iq][j] * t3[i];
                A[i][j] += acc0;
            }
        }
    }
}