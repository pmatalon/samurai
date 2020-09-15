#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"


// We use the Geier scheme because
// it seems to work pretty well


const double lambda = 1.; // Lattice velocity of the scheme
const double rho0 = 1.; // Reference density
const double u0 = 0.05; // Reference x-velocity
const double mu = 5.e-6; // Bulk viscosity
const double zeta = 100. * mu; // Shear viscosity

// The relaxation parameters will be computed in the sequel
// because they depend on the space step of the scheme


template<class Config>
auto init_f(mure::Mesh<Config> &mesh)
{
    constexpr std::size_t nvel = 9;
    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0},
                       {mure::BCType::neumann, 0.0}
                    }} };

    mure::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double rho = rho0;
        double qx = 0.;
        // double qx = u0 * std::exp(-400. * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5)));
        double qy = 0.;


        double r1 = 1.0 / lambda;
        double r2 = 1.0 / (lambda*lambda);
        double r3 = 1.0 / (lambda*lambda*lambda);
        double r4 = 1.0 / (lambda*lambda*lambda*lambda);


        // This is the Geier choice of momenti

        double cs2 = (lambda*lambda)/ 3.0; // Sound velocity of the lattice squared

        double m0 = rho;
        double m1 = qx;
        double m2 = qy;
        double m3 = (qx*qx+qy*qy)/rho + 2.*rho*cs2;
        double m4 = qx*(cs2+(qy/rho)*(qy/rho));
        double m5 = qy*(cs2+(qx/rho)*(qx/rho));
        double m6 = rho*(cs2+(qx/rho)*(qx/rho))*(cs2+(qy/rho)*(qy/rho));
        double m7 = (qx*qx-qy*qy)/rho;
        double m8 = qx*qy/rho;

        // We come back to the distributions



        f[cell][0] = m0                      -     r2*m3                        +     r4*m6                         ;
        f[cell][1] =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
        f[cell][2] =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
        f[cell][3] =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
        f[cell][4] =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
        f[cell][5] =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
        f[cell][6] =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
        f[cell][7] =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
        f[cell][8] =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;



        // // This is the Lallemand choice of momenti

        // double d = 1./rho0;

        // double m0 = rho;
        // double m1 = qx;
        // double m2 = qy;
        // double m3 = d * (qx*qx + qy*qy) + 2./3.*lambda*lambda*rho;
        // double m4 = 4./3.*lambda*lambda*qx;
        // double m5 = 4./3.*lambda*lambda*qy;
        // double m6 = 1./9. * (lambda*lambda*(21.*d*(qx*qx + qy*qy)+6.*lambda*lambda*rho)
        //                      -(6.*d*(qx*qx + qy*qy) - 2.*lambda*lambda*rho));
        // double m7 = d*(qx*qx - qy*qy);
        // double m8 = d*qx*qy;

        // f[cell][0] = m0                        - 1.5*r2*m3                          +  .5*r4*m6                         ;
        // f[cell][1] =      1.*r1*m1            +   .5*r2*m3 -  .5*r3*m4              - .25*r4*m6 + .25*r2*m7             ;
        // f[cell][2] =                 1.*r1*m2 +   .5*r2*m3             -  .5*r3*m5  - .25*r4*m6 - .25*r2*m7             ;
        // f[cell][3] =     -1.*r1*m1            +   .5*r2*m3 +  .5*r3*m4              - .25*r4*m6 + .25*r2*m7             ;
        // f[cell][4] =               - 1.*r1*m2 +   .5*r2*m3             +  .5*r3*m5  - .25*r4*m6 - .25*r2*m7             ;
        // f[cell][5] =    -.25*r1*m1 -.25*r1*m2 - .125*r2*m3 + .25*r3*m4 + .25*r3*m5 + .125*r4*m6             + .25*r2*m8 ;
        // f[cell][6] =     .25*r1*m1 -.25*r1*m2 - .125*r2*m3 - .25*r3*m4 + .25*r3*m5 + .125*r4*m6             - .25*r2*m8 ;                                     
        // f[cell][7] =     .25*r1*m1 +.25*r1*m2 - .125*r2*m3 - .25*r3*m4 - .25*r3*m5 + .125*r4*m6             + .25*r2*m8 ;                                     
        // f[cell][8] =    -.25*r1*m1 +.25*r1*m2 - .125*r2*m3 + .25*r3*m4 - .25*r3*m5 + .125*r4*m6             - .25*r2*m8 ;                                     


        
    });

    return f;
}


template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);


    auto rotation_of_pi_over_two = [] (int alpha, int k, int h)
    {
        // Returns the rotation of (k, h) of an angle alpha * pi / 2.
        // All the operations are performed on integer, to be exact

        int cosinus = static_cast<int>(std::round(std::cos(alpha * M_PI / 2.)));
        int sinus   = static_cast<int>(std::round(std::sin(alpha * M_PI / 2.)));

        return std::pair<int, int> (cosinus * k - sinus   * h, 
                                      sinus * k + cosinus * h);
    };

    // Transforms the coordinates to apply the rotation
    auto tau = [] (int delta, int k)
    {
        // The case in which delta = 0 is rather exceptional
        if (delta == 0) {
            return k;
        }
        else {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < tmp) ? (k - tmp) : (k - tmp + 1));
        }
    };

    auto tau_inverse = [] (int delta, int k)
    {
        if (delta == 0) {
            return k;
        }
        else
        {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < 0) ? (k + tmp) : (k + tmp - 1));   
        }
    };

    for(std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1<<k);

        // We have 9 velocity out of which 8 are moving
        // 4 are moving along the axis, thus needing only 2 fluxes each (entering-exiting)
        // and 4 along the diagonals, thus needing  6 fluxes

        // 4 * 2 + 4 * 6 = 8 + 24 = 32 

        data[k].resize(32);


        // Parallel velocities
        
        for (int alpha = 0; alpha <= 3; ++alpha)
        {

            // std::cout<<std::endl<<"Testing rotations - alpha = "<<alpha<<std::endl;
            // auto rot_1_0 = rotation_of_pi_over_two(alpha, 1, 0);
            // auto rot_0_1 = rotation_of_pi_over_two(alpha, 0, 1);

            // std::cout<<rot_1_0.first<<"  "<<rot_1_0.second<<"\n"<<rot_0_1.first<<"  "<<rot_0_1.second<<std::endl;
            
            for (int l = 0; l < size; ++l)
            {
                // The reference direction from which the other ones are computed
                // is that of (1, 0)
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));

                data[k][0 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_in.first ), tau_inverse(k, rotated_in.second ));
                data[k][1 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));

            }
        }

        // Diagonal velocities

        // Translation of the indices from which we start saving
        // the new computations
        int offset = 4 * 2;

        for (int alpha = 0; alpha <= 3; ++alpha)
        {

            // First side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));
                
                data[k][offset + 6 * alpha + 0] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 3] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));
   
            }

            // Cell on the diagonal
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i    * size - 1), tau(k,  j    * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1) * size - 1), tau(k, (j+1) * size - 1));

                data[k][offset + 6 * alpha + 1] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 4] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));
   
            }
            // Second side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha,  tau(k, i*size + l), tau(k,  j    * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha,  tau(k, i*size + l), tau(k, (j+1) * size - 1));
                
                data[k][offset + 6 * alpha + 2] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 5] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));                
            }

        }
    }
    return data;
}

// I do many separate functions because the return type
// is not necessarely the same between directions and I want to avoid 
// using a template, which indeed comes back to the same than this.
template<class Mesh> 
auto get_adjacent_boundary_east(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level); // When we are not at the finest level, we must translate more

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp))), // Removing SE
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_north(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp))), // Removing NW
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_west(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp))), // Removing NW
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp))), // Removing SW
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_south(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp))), // Removing SE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp))), // Removing SW
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_northeast(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * d11)),
                                              translate(mesh.initial_mesh(), - coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), - coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_northwest(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * d1m1)),
                                              translate(mesh.initial_mesh(), - coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_southwest(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * d11)),
                                              translate(mesh.initial_mesh(), coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh> 
auto get_adjacent_boundary_southeast(Mesh & mesh, std::size_t level, mure::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * d1m1)),
                                              translate(mesh.initial_mesh(), coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), -coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}





// We have to average only the fluxes
template<class Field, class pred>
void one_time_step_overleaves_corrected(Field &f, const pred& pred_coeff, std::size_t iter)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();
    
    std::cout<<std::endl<<"[+] Projecting"<<std::flush;
    mure::mr_projection(f);
    std::cout<<std::endl<<"[+] Updating BC"<<std::flush;
    f.update_bc(); // It is important to do so
    std::cout<<std::endl<<"[+] Predicting"<<std::flush;
    mure::mr_prediction(f);
    std::cout<<std::endl<<"[+] Predicting overleaves"<<std::flush;
    mure::mr_prediction_overleaves(f);


    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    // This stored the fluxes computed at the level
    // of the overleaves
    Field fluxes{"fluxes", mesh};
    fluxes.array().fill(0.);


    Field advected{"advected", mesh};
    advected.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        if (level == max_level) {



            std::cout<<std::endl<<"[+] Advecting at finest"<<std::flush;
            
            std::cout<<std::endl<<"[=] East"<<std::flush;
            auto leaves_east = get_adjacent_boundary_east(mesh, max_level, mure::MeshType::cells);
            leaves_east.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                // // BC
                advected(3, level, k, h) =  f(1, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);

                

            });
            
            std::cout<<std::endl<<"[=] North"<<std::flush;
            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, mure::MeshType::cells);
            leaves_north.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 


            double rho = rho0;
            double qx = rho0*u0;
            double qy = 0.;


            double r1 = 1.0 / lambda;
            double r2 = 1.0 / (lambda*lambda);
            double r3 = 1.0 / (lambda*lambda*lambda);
            double r4 = 1.0 / (lambda*lambda*lambda*lambda);


            // This is the Geier choice of momenti

            double cs2 = (lambda*lambda)/ 3.0; // Sound velocity of the lattice squared

            double m0 = rho;
            double m1 = qx;
            double m2 = qy;
            double m3 = (qx*qx+qy*qy)/rho + 2.*rho*cs2;
            double m4 = qx*(cs2+(qy/rho)*(qy/rho));
            double m5 = qy*(cs2+(qx/rho)*(qx/rho));
            double m6 = rho*(cs2+(qx/rho)*(qx/rho))*(cs2+(qy/rho)*(qy/rho));
            double m7 = (qx*qx-qy*qy)/rho;
            double m8 = qx*qy/rho;

            // We come back to the distributions

            advected(0, level, k, h) = m0                      -     r2*m3                        +     r4*m6                         ;
            advected(1, level, k, h) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
            advected(2, level, k, h) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
            advected(3, level, k, h) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
            advected(4, level, k, h) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
            advected(5, level, k, h) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
            advected(6, level, k, h) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
            advected(7, level, k, h) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
            advected(8, level, k, h) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;



            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, mure::MeshType::cells);
            leaves_northeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 


                auto rho = f(0, level, k, h) + f(1, level, k, h) + f(2, level, k, h) + f(3, level, k, h) + f(4, level, k, h)
                                             + f(5, level, k, h) + f(6, level, k, h) + f(7, level, k, h) + f(8, level, k, h);

                advected(4, level, k, h) =  f(2, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);
                advected(3, level, k, h) =  f(3, level, k, h);
                advected(6, level, k, h) =  f(6, level, k, h);
                
                

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);


            });

            std::cout<<std::endl<<"[=] West"<<std::flush;
            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, mure::MeshType::cells);
            leaves_west.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                advected(1, level, k, h) =  f(3, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);


            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, mure::MeshType::cells);
            leaves_northwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 


                auto rho = f(0, level, k, h) + f(1, level, k, h) + f(2, level, k, h) + f(3, level, k, h) + f(4, level, k, h)
                                             + f(5, level, k, h) + f(6, level, k, h) + f(7, level, k, h) + f(8, level, k, h);

                advected(1, level, k, h) =  f(3, level, k, h);
                advected(4, level, k, h) =  f(2, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);
                

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);

            });

            std::cout<<std::endl<<"[=] South"<<std::flush;
            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, mure::MeshType::cells);
            leaves_south.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                advected(2, level, k, h) =  f(4, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);


            });

            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, mure::MeshType::cells);
            leaves_southwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                

                advected(2, level, k, h) =  f(4, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);
                advected(1, level, k, h) =  f(3, level, k, h);
                advected(8, level, k, h) =  f(6, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);

            });

            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, mure::MeshType::cells);
            leaves_southeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                advected(2, level, k, h) =  f(4, level, k, h);
                advected(5, level, k, h) =  f(7, level, k, h);
                advected(6, level, k, h) =  f(8, level, k, h);
                advected(3, level, k, h) =  f(1, level, k, h);
                advected(7, level, k, h) =  f(5, level, k, h);


                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);
            });


            // Advection far from the boundary
            auto tmp1 = union_(union_(union_(leaves_east, leaves_north), leaves_west), leaves_south);
            auto tmp2 = union_(union_(union_(leaves_northeast, leaves_northwest), leaves_southwest), leaves_southeast);
            auto all_leaves_boundary = union_(tmp1, tmp2);
            auto internal_leaves = mure::difference(mesh[mure::MeshType::cells][max_level],
                                      all_leaves_boundary).on(max_level); // It is very important to project at this point

            std::cout<<std::endl<<"[=] Far from the boundary"<<std::flush;
            internal_leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                advected(0, level, k, h) =  f(0, level, k, h);
                advected(1, level, k, h) =  f(1, level, k - 1, h);
                advected(2, level, k, h) =  f(2, level, k,     h - 1);
                advected(3, level, k, h) =  f(3, level, k + 1, h);
                advected(4, level, k, h) =  f(4, level, k,     h + 1);
                advected(5, level, k, h) =  f(5, level, k - 1, h - 1);
                advected(6, level, k, h) =  f(6, level, k + 1, h - 1);
                advected(7, level, k, h) =  f(7, level, k + 1, h + 1);
                advected(8, level, k, h) =  f(8, level, k - 1, h + 1);
            });

            

            // Its time for collision which is local
            auto leaves = intersection(mesh[mure::MeshType::cells][max_level], 
                                       mesh[mure::MeshType::cells][max_level]);

            std::cout<<std::endl<<"[+] Colliding at finest"<<std::flush;
            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y  

                auto f0 = xt::eval(advected(0, level, k, h));
                auto f1 = xt::eval(advected(1, level, k, h));
                auto f2 = xt::eval(advected(2, level, k, h));
                auto f3 = xt::eval(advected(3, level, k, h));
                auto f4 = xt::eval(advected(4, level, k, h));
                auto f5 = xt::eval(advected(5, level, k, h));
                auto f6 = xt::eval(advected(6, level, k, h));
                auto f7 = xt::eval(advected(7, level, k, h));
                auto f8 = xt::eval(advected(8, level, k, h));
                
              
                // // We compute the advected momenti
                double l1 = lambda;
                double l2 = l1 * lambda;
                double l3 = l2 * lambda;
                double l4 = l3 * lambda;

                double r1 = 1.0 / lambda;
                double r2 = 1.0 / (lambda*lambda);
                double r3 = 1.0 / (lambda*lambda*lambda);
                double r4 = 1.0 / (lambda*lambda*lambda*lambda);



                // // Choice of momenti by Geier

                auto m0 = xt::eval(    f0 + f1 + f2 + f3 + f4 +   f5 +   f6 +   f7 +   f8 ) ;
                auto m1 = xt::eval(l1*(     f1      - f3      +   f5 -   f6 -   f7 +   f8 ) );
                auto m2 = xt::eval(l1*(          f2      - f4 +   f5 +   f6 -   f7 -   f8 ) );
                auto m3 = xt::eval(l2*(     f1 + f2 + f3 + f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8 ) );
                auto m4 = xt::eval(l3*(                           f5 -   f6 -   f7 +   f8 ) );
                auto m5 = xt::eval(l3*(                           f5 +   f6 -   f7 -   f8 ) );
                auto m6 = xt::eval(l4*(                           f5 +   f6 +   f7 +   f8 ) );
                auto m7 = xt::eval(l2*(     f1 - f2 + f3 - f4                             ) );
                auto m8 = xt::eval(l2*(                           f5 -   f6 +   f7 -   f8 ) );

                // Collision

                double space_step = 1.0 / (1 << max_level);
                double dummy = 3.0/(lambda*rho0*space_step);
                double sigma_1 = dummy*(zeta - 2.*mu/3.);
                double sigma_2 = dummy*mu;
                double s_1 = 1/(.5+sigma_1);
                double s_2 = 1/(.5+sigma_2);

                double cs2 = (lambda * lambda) / 3.0; // sound velocity squared

                m3 = (1. - s_1) * m3 + s_1 * ((m1*m1+m2*m2)/m0 + 2.*m0*cs2);
                m4 = (1. - s_1) * m4 + s_1 * (m1*(cs2+(m2/m0)*(m2/m0)));
                m5 = (1. - s_1) * m5 + s_1 * (m2*(cs2+(m1/m0)*(m1/m0)));
                m6 = (1. - s_1) * m6 + s_1 * (m0*(cs2+(m1/m0)*(m1/m0))*(cs2+(m2/m0)*(m2/m0)));
                m7 = (1. - s_2) * m7 + s_2 * ((m1*m1-m2*m2)/m0);
                m8 = (1. - s_2) * m8 + s_2 * (m1*m2/m0);


                // // We come back to the distributions

  

                new_f(0, level, k, h) = m0                      -     r2*m3                        +     r4*m6                           ;
                new_f(1, level, k, h) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
                new_f(2, level, k, h) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
                new_f(3, level, k, h) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
                new_f(4, level, k, h) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
                new_f(5, level, k, h) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(6, level, k, h) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
                new_f(7, level, k, h) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(8, level, k, h) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;

                // // Choice of momenti by Lallemand

                // auto m0 = xt::eval(    f0 + f1 + f2 + f3 + f4 +   f5 +   f6 +   f7 +   f8 ) ;
                // auto m1 = xt::eval(l1*(     f1      - f3      +   f5 -   f6 -   f7 +   f8 ) );
                // auto m2 = xt::eval(l1*(          f2      - f4 +   f5 +   f6 -   f7 -   f8 ) );
                // auto m3 = xt::eval(l2*(     f1 + f2 + f3 + f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8 ) );
                // auto m4 = xt::eval(l3*(     f1      - f3      + 2*f5 - 2*f6 - 2*f7 + 2*f8 ) );
                // auto m5 = xt::eval(l3*(          f2      - f4 + 2*f5 + 2*f6 - 2*f7 - 2*f8 ) );
                // auto m6 = xt::eval(l4*(     f1 + f2 + f3 + f4 + 4*f5 + 4*f6 + 4*f7 + 4*f8 ) );
                // auto m7 = xt::eval(l2*(     f1 - f2 + f3 - f4                             ) );
                // auto m8 = xt::eval(l2*(                           f5 -   f6 +   f7 -   f8 ) );

                // double space_step = 1.0 / (1 << max_level);
                // double dummy = 3./(lambda * space_step * rho0);
                // double sigma1 = dummy * zeta;
                // double sigma2 = dummy * mu;

                // double s1 = 1./(0.5 + sigma1);
                // double s2 = 1./(0.5 + sigma2);

                // double d = 1./rho0;

                // m3 = (1. - s1) * m3 + s1 * (d * (m1*m1 + m2*m2) + 2./3.*lambda*lambda*m0);
                // m4 = (1. - s1) * m4 + s1 * (4./3.*lambda*lambda*m1);
                // m5 = (1. - s1) * m5 + s1 * (4./3.*lambda*lambda*m2);
                // m6 = (1. - s1) * m6 + s1 * (1./9. * (lambda*lambda*(21.*d*(m1*m1 + m2*m2)+6.*lambda*lambda*m0) - (6.*d*(m1*m1 + m2*m2) - 2.*lambda*lambda*m0)));
                // m7 = (1. - s2) * m7 + s2 * (d*(m1*m1 - m2*m2));
                // m8 = (1. - s2) * m8 + s2 * (d*m1*m2);

                // new_f(0, level, k, h) = m0                        - 1.5*r2*m3                          +  .5*r4*m6                         ;
                // new_f(1, level, k, h) =      1.*r1*m1            +   .5*r2*m3 -  .5*r3*m4              - .25*r4*m6 + .25*r2*m7             ;
                // new_f(2, level, k, h) =                 1.*r1*m2 +   .5*r2*m3             -  .5*r3*m5  - .25*r4*m6 - .25*r2*m7             ;
                // new_f(3, level, k, h) =     -1.*r1*m1            +   .5*r2*m3 +  .5*r3*m4              - .25*r4*m6 + .25*r2*m7             ;
                // new_f(4, level, k, h) =               - 1.*r1*m2 +   .5*r2*m3             +  .5*r3*m5  - .25*r4*m6 - .25*r2*m7             ;
                // new_f(5, level, k, h) =    -.25*r1*m1 -.25*r1*m2 - .125*r2*m3 + .25*r3*m4 + .25*r3*m5 + .125*r4*m6             + .25*r2*m8 ;
                // new_f(6, level, k, h) =     .25*r1*m1 -.25*r1*m2 - .125*r2*m3 - .25*r3*m4 + .25*r3*m5 + .125*r4*m6             - .25*r2*m8 ;                                     
                // new_f(7, level, k, h) =     .25*r1*m1 +.25*r1*m2 - .125*r2*m3 - .25*r3*m4 - .25*r3*m5 + .125*r4*m6             + .25*r2*m8 ;                                     
                // new_f(8, level, k, h) =    -.25*r1*m1 +.25*r1*m2 - .125*r2*m3 + .25*r3*m4 - .25*r3*m5 + .125*r4*m6             - .25*r2*m8 ;  

            });                       


        }


        else
        {
            std::size_t j = max_level - (level + 1);
            double coeff = 1. / (1 << (2*j)); // ATTENTION A LA DIMENSION 2 !!!!

            std::cout<<std::endl<<"[+] Advecting at level "<<level<<" with overleaves at "<<(level + 1)<<std::flush;


            // Touching west

            auto overleaves_west = intersection(get_adjacent_boundary_west(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_northwest = intersection(get_adjacent_boundary_northwest(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
            auto overleaves_southwest = intersection(get_adjacent_boundary_southwest(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                                                    
            auto touching_west = union_(union_(overleaves_west, overleaves_northwest), 
                                        overleaves_southwest);


            std::cout<<std::endl<<"[=] West/NorthWest/SouthWest"<<std::flush;
            touching_west.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                std::vector<int> flx_num {4, 16, 20};
                std::vector<int> flx_vel {3, 6,  7};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            }); 

            std::cout<<std::endl<<"[=] West"<<std::flush;
            overleaves_west.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {2, 6, 10, 14, 15, 21, 22, 26};
                std::vector<int> flx_vel {2, 4, 5,  6,  6,  7,  7,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
            
                // Bounce back to velocity (0, 0)
                // h = 1
                fluxes(1, level + 1, k, h) += (1 << j) * coeff * f(3, level + 1, k, h);
                // h = 5
                fluxes(5, level + 1, k, h) += (1 << j) * coeff * f(7, level + 1, k, h);
                // h = 8
                fluxes(8, level + 1, k, h) += (1 << j) * coeff * f(6, level + 1, k, h);
            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            overleaves_northwest.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {2, 10, 14, 15};
                std::vector<int> flx_vel {2, 5,  6,  6 };

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
            
                // Bounce back to velocity (0, 0)
                // h = 1
                fluxes(1, level + 1, k, h) += (1 << j) * coeff * f(3, level + 1, k, h);
                // h = 4
                fluxes(4, level + 1, k, h) += (1 << j) * coeff * f(2, level + 1, k, h);
                // h = 5
                fluxes(5, level + 1, k, h) += (1 << j) * coeff * f(7, level + 1, k, h);
                // h = 7
                fluxes(7, level + 1, k, h) += (1 << j) * coeff * f(5, level + 1, k, h);
                // h = 8
                fluxes(8, level + 1, k, h) += (2 * (1 << j) - 1) * coeff * f(6, level + 1, k, h);
            });



            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            overleaves_southwest.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {6, 21, 22, 26};
                std::vector<int> flx_vel {4, 7,  7,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
            
                // Bounce back to velocity (0, 0)
                // h = 1
                fluxes(1, level + 1, k, h) += (1 << j) * coeff * f(3, level + 1, k, h);
                // h = 2
                fluxes(2, level + 1, k, h) += (1 << j) * coeff * f(4, level + 1, k, h);
                // h = 5
                fluxes(5, level + 1, k, h) += (2 * (1 << j) - 1) * coeff * f(7, level + 1, k, h);
                // h = 6
                fluxes(6, level + 1, k, h) += (1 << j) * coeff * f(8, level + 1, k, h);
                // h = 8
                fluxes(8, level + 1, k, h) += (1 << j) * coeff * f(6, level + 1, k, h);
            });

















            // This is necessary because the only overleaves we have to advect
            // on are the ones superposed with the leaves to which we come back
            // eventually in the process
            auto overleaves_east = intersection(get_adjacent_boundary_east(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_northeast = intersection(get_adjacent_boundary_northeast(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                
            auto overleaves_southeast = intersection(get_adjacent_boundary_southeast(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 
                                                                                    
            auto touching_east = union_(union_(overleaves_east, overleaves_northeast), 
                                        overleaves_southeast);



            std::cout<<std::endl<<"[=] East/NorthEast/SouthEast"<<std::flush;
            touching_east.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                std::vector<int> flx_num {0, 8, 28};
                std::vector<int> flx_vel {1, 5, 8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });

            std::cout<<std::endl<<"[=] East"<<std::flush;
            overleaves_east.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {2, 6, 9, 10, 14, 22, 26, 27};
                std::vector<int> flx_vel {2, 4, 5, 5,  6,  7,  8,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
            
                // Bounce back to velocity (0, 0)
                // h = 3
                fluxes(3, level + 1, k, h) += (1 << j) * coeff * f(1, level + 1, k, h);
                // h = 6
                fluxes(6, level + 1, k, h) += (1 << j) * coeff * f(8, level + 1, k, h);
                // h = 7
                fluxes(7, level + 1, k, h) += (1 << j) * coeff * f(5, level + 1, k, h);
            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            overleaves_northeast.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {2, 9, 10, 14};
                std::vector<int> flx_vel {2, 5, 5,  6 };     


                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   

                // Bounce back to velocity (0, 0) and moving lid as well
                auto rho = f(0, level + 1, k, h) + f(1, level + 1, k, h) + f(2, level + 1, k, h) + f(3, level + 1, k, h) + f(4, level + 1, k, h)
                                                 + f(5, level + 1, k, h) + f(6, level + 1, k, h) + f(7, level + 1, k, h) + f(8, level + 1, k, h);


                // h = 3
                fluxes(3, level + 1, k, h) += (1 << j) * coeff * f(1, level + 1, k, h);
                // h = 4
                fluxes(4, level + 1, k, h) += (1 << j) * coeff * f(2, level + 1, k, h);
                // h = 6
                fluxes(6, level + 1, k, h) += (1 << j) * coeff * f(8, level + 1, k, h);
                // h = 7
                fluxes(7, level + 1, k, h) += (1 << j) * coeff * (f(5, level + 1, k, h) - 0.5 * u0 * rho);
                fluxes(7, level + 1, k, h) += ((1 << j) - 1) * coeff * f(5, level + 1, k, h);
                // h = 8
                fluxes(8, level + 1, k, h) += (1 << j) * coeff * (f(6, level + 1, k, h) + 0.5 * u0 * rho);

            });



            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            overleaves_southeast.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {6, 22, 26, 27};
                std::vector<int> flx_vel {4, 7,  8,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
           
                // Bounce back to velocity (0, 0)
                // h = 2
                fluxes(2, level + 1, k, h) += (1 << j) * coeff * f(4, level + 1, k, h);
                // h = 3
                fluxes(3, level + 1, k, h) += (1 << j) * coeff * f(1, level + 1, k, h);
                // h = 5
                fluxes(5, level + 1, k, h) += (1 << j) * coeff * f(7, level + 1, k, h);
                // h = 6
                fluxes(6, level + 1, k, h) += (2 * (1 << j) - 1) * coeff * f(8, level + 1, k, h);
                // h = 7
                fluxes(7, level + 1, k, h) += (1 << j) * coeff * f(5, level + 1, k, h);
            });






            auto overleaves_south = intersection(get_adjacent_boundary_south(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto overleaves_north = intersection(get_adjacent_boundary_north(mesh, level + 1, mure::MeshType::overleaves), 
                                                mesh[mure::MeshType::cells][level]); 

            auto north_and_south = union_(overleaves_south, overleaves_north);


            std::cout<<std::endl<<"[=] North"<<std::flush;
            overleaves_north.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {0, 2, 4, 8, 9, 10, 14, 15, 16, 20, 28};
                std::vector<int> flx_vel {1, 2, 3, 5, 5, 5,  6,  6,  6,  7,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
            
                // Bounce back to velocity of the top lid
                auto rho = f(0, level + 1, k, h) + f(1, level + 1, k, h) + f(2, level + 1, k, h) + f(3, level + 1, k, h) + f(4, level + 1, k, h)
                                                 + f(5, level + 1, k, h) + f(6, level + 1, k, h) + f(7, level + 1, k, h) + f(8, level + 1, k, h);

                // h = 4
                fluxes(4, level + 1, k, h) += (1 << j) * coeff * f(2, level + 1, k, h);
                // h = 7
                fluxes(7, level + 1, k, h) += (1 << j) * coeff * (f(5, level + 1, k, h) - 0.5 * u0 * rho);
                // h = 8
                fluxes(8, level + 1, k, h) += (1 << j) * coeff * (f(6, level + 1, k, h) + 0.5 * u0 * rho);
            });



            std::cout<<std::endl<<"[=] South"<<std::flush;
            overleaves_south.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Regular
                std::vector<int> flx_num {0, 4, 6, 8, 16, 20, 21, 22, 26, 27, 28};
                std::vector<int> flx_vel {1, 3, 4, 5, 6,  7,  7,  7,  8,  8,  8};


                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                }         

                // To correct   
            
                // Bounce back to (0, 0)
                // h = 2
                fluxes(2, level + 1, k, h) += (1 << j) * coeff * f(4, level + 1, k, h);
                // h = 5
                fluxes(5, level + 1, k, h) += (1 << j) * coeff * f(7, level + 1, k, h);
                // h = 6
                fluxes(6, level + 1, k, h) += (1 << j) * coeff * f(8, level + 1, k, h);

            });



            // All the exiting fluxes are valid, thus we perform them
            // once for all

            auto all_overleaves = intersection(mesh[mure::MeshType::cells][level], 
                                               mesh[mure::MeshType::cells][level]).on(level + 1);  

            std::cout<<std::endl<<"[=] All the exiting fluxes"<<std::flush;
            all_overleaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                                
                std::vector<int> flx_num {1, 3, 5, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 29, 30, 31};
                std::vector<int> flx_vel {1, 2, 3, 4, 5,  5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        // Be careful about the - sign because we are dealing 
                        // with exiting fluxes
                        fluxes(flx_vel[idx], level + 1, k, h) -= coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                } 

            });                              

            std::cout<<std::endl<<"[=] Far from the boundary"<<std::flush;
            auto overleaves_far_boundary = difference(mesh[mure::MeshType::cells][level], 
                                                      union_(union_(touching_east, touching_west), 
                                                             north_and_south)).on(level + 1);  // Again, it is very important to project before using

            // We are just left to add the incoming fluxes to to the internal overleaves
            overleaves_far_boundary([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 
                
                std::vector<int> flx_num {0, 2, 4, 6, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28};
                std::vector<int> flx_vel {1, 2, 3, 4, 5, 5, 5,  6,  6,  6,  7,  7,  7,  8,  8,  8};

                for (int idx = 0; idx < flx_num.size(); ++idx)  {
                    for(auto &c: pred_coeff[j][flx_num[idx]].coeff)
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                    }
                } 

            });

            auto leaves = mure::intersection(mesh[mure::MeshType::cells][level],
                                             mesh[mure::MeshType::cells][level]);

            std::cout<<std::endl<<"[+] Projection of the overleaves on their leaves and collision"<<std::flush;
            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y 

                auto f0 = xt::eval(f(0, level, k, h));

                auto f1 = xt::eval(f(1, level, k, h)) + 0.25 * (fluxes(1, level + 1, 2*k,     2*h) 
                                                              + fluxes(1, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(1, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(1, level + 1, 2*k + 1, 2*h + 1));

                auto f2 = xt::eval(f(2, level, k, h)) + 0.25 * (fluxes(2, level + 1, 2*k,     2*h) 
                                                              + fluxes(2, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(2, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(2, level + 1, 2*k + 1, 2*h + 1));

                auto f3 = xt::eval(f(3, level, k, h)) + 0.25 * (fluxes(3, level + 1, 2*k,     2*h) 
                                                              + fluxes(3, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(3, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(3, level + 1, 2*k + 1, 2*h + 1));

                auto f4 = xt::eval(f(4, level, k, h)) + 0.25 * (fluxes(4, level + 1, 2*k,     2*h) 
                                                              + fluxes(4, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(4, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(4, level + 1, 2*k + 1, 2*h + 1));

                auto f5 = xt::eval(f(5, level, k, h)) + 0.25 * (fluxes(5, level + 1, 2*k,     2*h) 
                                                              + fluxes(5, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(5, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(5, level + 1, 2*k + 1, 2*h + 1));

                auto f6 = xt::eval(f(6, level, k, h)) + 0.25 * (fluxes(6, level + 1, 2*k,     2*h) 
                                                              + fluxes(6, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(6, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(6, level + 1, 2*k + 1, 2*h + 1));

                auto f7 = xt::eval(f(7, level, k, h)) + 0.25 * (fluxes(7, level + 1, 2*k,     2*h) 
                                                              + fluxes(7, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(7, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(7, level + 1, 2*k + 1, 2*h + 1));

                auto f8 = xt::eval(f(8, level, k, h)) + 0.25 * (fluxes(8, level + 1, 2*k,     2*h) 
                                                              + fluxes(8, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(8, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(8, level + 1, 2*k + 1, 2*h + 1));

                auto f9 = xt::eval(f(9, level, k, h)) + 0.25 * (fluxes(9, level + 1, 2*k,     2*h) 
                                                              + fluxes(9, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(9, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(9, level + 1, 2*k + 1, 2*h + 1));



                // // We compute the advected momenti
                double l1 = lambda;
                double l2 = l1 * lambda;
                double l3 = l2 * lambda;
                double l4 = l3 * lambda;

                auto m0 = xt::eval(    f0 + f1 + f2 + f3 + f4 +   f5 +   f6 +   f7 +   f8 ) ;
                auto m1 = xt::eval(l1*(     f1      - f3      +   f5 -   f6 -   f7 +   f8 ) );
                auto m2 = xt::eval(l1*(          f2      - f4 +   f5 +   f6 -   f7 -   f8 ) );
                auto m3 = xt::eval(l2*(     f1 + f2 + f3 + f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8 ) );
                auto m4 = xt::eval(l3*(                           f5 -   f6 -   f7 +   f8 ) );
                auto m5 = xt::eval(l3*(                           f5 +   f6 -   f7 -   f8 ) );
                auto m6 = xt::eval(l4*(                           f5 +   f6 +   f7 +   f8 ) );
                auto m7 = xt::eval(l2*(     f1 - f2 + f3 - f4                             ) );
                auto m8 = xt::eval(l2*(                           f5 -   f6 +   f7 -   f8 ) );

                // Collision
                double space_step = 1.0 / (1 << max_level);
                double dummy = 3.0/(lambda*rho0*space_step);
                double sigma_1 = dummy*(zeta - 2.*mu/3.);
                double sigma_2 = dummy*mu;
                double s_1 = 1/(.5+sigma_1);
                double s_2 = 1/(.5+sigma_2);

                double cs2 = (lambda * lambda) / 3.0; // sound velocity squared

                m3 = (1. - s_1) * m3 + s_1 * ((m1*m1+m2*m2)/m0 + 2.*m0*cs2);
                m4 = (1. - s_1) * m4 + s_1 * (m1*(cs2+(m2/m0)*(m2/m0)));
                m5 = (1. - s_1) * m5 + s_1 * (m2*(cs2+(m1/m0)*(m1/m0)));
                m6 = (1. - s_1) * m6 + s_1 * (m0*(cs2+(m1/m0)*(m1/m0))*(cs2+(m2/m0)*(m2/m0)));
                m7 = (1. - s_2) * m7 + s_2 * ((m1*m1-m2*m2)/m0);
                m8 = (1. - s_2) * m8 + s_2 * (m1*m2/m0);

                // We come back to the distributions

                double r1 = 1.0 / lambda;
                double r2 = 1.0 / (lambda*lambda);
                double r3 = 1.0 / (lambda*lambda*lambda);
                double r4 = 1.0 / (lambda*lambda*lambda*lambda);


                new_f(0, level, k, h) = m0                      -     r2*m3                        +     r4*m6                           ;
                new_f(1, level, k, h) =     .5*r1*m1            + .25*r2*m3 - .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
                new_f(2, level, k, h) =                .5*r1*m2 + .25*r2*m3            -  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
                new_f(3, level, k, h) =    -.5*r1*m1            + .25*r2*m3 + .5*r3*m4             -  .5*r4*m6 + .25*r2*m7             ;
                new_f(4, level, k, h) =              - .5*r1*m2 + .25*r2*m3            +  .5*r3*m5 -  .5*r4*m6 - .25*r2*m7             ;
                new_f(5, level, k, h) =                                      .25*r3*m4 + .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(6, level, k, h) =                                     -.25*r3*m4 + .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
                new_f(7, level, k, h) =                                     -.25*r3*m4 - .25*r3*m5 + .25*r4*m6             + .25*r2*m8 ;
                new_f(8, level, k, h) =                                      .25*r3*m4 - .25*r3*m5 + .25*r4*m6             - .25*r2*m8 ;
            });

        }
        
    }

    std::swap(f.array(), new_f.array());
}




template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q9_Lid_Driven_Cavity_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = mure::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    mure::Field<Config> level_{"level", mesh};
    mure::Field<Config> rho{"rho", mesh};
    mure::Field<Config> qx{"qx", mesh};
    mure::Field<Config> qy{"qy", mesh};
    mure::Field<Config> vel_mod{"vel_modulus", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4] 
                               + f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];

        qx[cell] = lambda * (f[cell][1] - f[cell][3] + f[cell][5] - f[cell][6] - f[cell][7] + f[cell][8]);
        qy[cell] = lambda * (f[cell][2] - f[cell][4] + f[cell][5] + f[cell][6] - f[cell][7] - f[cell][8]);

        vel_mod[cell] = xt::sqrt(qx[cell] * qx[cell] 
                               + qy[cell] * qy[cell]) / rho[cell];

    });


    h5file.add_field(rho);
    h5file.add_field(qx);
    h5file.add_field(qy);
    h5file.add_field(vel_mod);

    h5file.add_field(f);
    h5file.add_field(level_);

}

int main(int argc, char *argv[])
{
    cxxopts::Options options("D2Q9 scheme for the simulation of the Von Karman vortex street",
                             "We will add multiresolution very soon");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("8"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("8"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {

            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 2;
            using Config = mure::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();


            mure::Box<double, dim> box({0, 0}, {1, 1});
            mure::Mesh<Config> mesh{box, min_level, max_level};



            using coord_index_t = typename Config::coord_index_t;
            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

            // std::cout<<std::endl<<"Showing prediction matrix for fluxes"<<std::endl;

            // for (int idx = 0; idx <= 31; ++idx){
            // std::cout<<std::endl<<"Idx = "<<idx<<std::endl;
            //             for (int k = 0; k <= max_level - min_level; ++k)
            // {
            //     for (auto cf : pred_coeff[k][idx].coeff){
            //         coord_index_t stencil_x, stencil_y;
            //         std::tie(stencil_x, stencil_y) = cf.first;
                    
                    
            //         std::cout<<"k = "<<k<<"  Offset x = "<<stencil_x<<"   Offset y = "<<stencil_y<<"   Value = "<<cf.second<<std::endl;
            //     }
                   
            // }
            // }


            // return 0;

            auto f = init_f(mesh);

            double T = 1000.;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout<<std::endl<<"Iteration number = "<<nb_ite<<std::endl;

                std::cout<<std::endl<<"[*] Coarsening"<<std::flush;
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (coarsening(f, eps, i))
                        break;
                }

                std::cout<<std::endl<<"[*] Refinement"<<std::flush;
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (refinement(f, eps, 0.0, i))
                        break;
                }

                std::cout<<std::endl<<"[*] Prediction overleaves before saving"<<std::flush;
                mure::mr_prediction_overleaves(f); // Before saving

                // f.update_bc(0);
                // std::stringstream str;
                // str << "D2Q9_lid_driven_debug_by_level_"<<nb_ite;
                // auto h5file = mure::Hdf5(str.str().data());
                // h5file.add_field_by_level(mesh, f);


                std::size_t howoften = 128;

                if (nb_ite % howoften == 0) {
                    std::cout<<std::endl<<"Saving"<<std::endl;
                    save_solution(f, eps, nb_ite/howoften, std::string("_before")); // Before applying the scheme
                }
                    

                one_time_step_overleaves_corrected(f, pred_coeff, nb_ite);
                
                // save_solution(f, eps, nb_ite/1, std::string("_after")); // Before applying the scheme

            }
    
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}