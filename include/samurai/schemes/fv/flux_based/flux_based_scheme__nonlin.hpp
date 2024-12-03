#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * @class FluxBasedScheme
     *    Implementation of non-linear schemes
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
      public:

        using base_class = FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;

        using typename base_class::field_value_type;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;

        using cell_t = typename mesh_t::cell_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

      private:

        FluxDefinition<cfg> m_flux_definition;
        bool m_include_boundary_fluxes = true;

      public:

        explicit FluxBasedScheme(const FluxDefinition<cfg>& flux_definition)
            : m_flux_definition(flux_definition)
        {
        }

        const auto& flux_definition() const
        {
            return m_flux_definition;
        }

        auto& flux_definition()
        {
            return m_flux_definition;
        }

        void include_boundary_fluxes(bool include)
        {
            m_include_boundary_fluxes = include;
        }

        bool include_boundary_fluxes() const
        {
            return m_include_boundary_fluxes;
        }

        inline auto h_factor(double h_face, double h_cell) const
        {
            double face_measure = std::pow(h_face, dim - 1);
            double cell_measure = std::pow(h_cell, dim);
            return face_measure / cell_measure;
        }

        template <class T> // FluxValue<cfg> or StencilJacobian<cfg>
        inline T contribution(const T& flux_value, double h_face, double h_cell) const
        {
            return h_factor(h_face, h_cell) * flux_value;
        }

        inline field_value_type flux_value_cmpnent(const FluxValue<cfg>& flux_value, [[maybe_unused]] std::size_t field_i) const
        {
            if constexpr (output_field_size == 1)
            {
                return flux_value;
            }
            else
            {
                return flux_value(field_i);
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the interior interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_interior_interface(std::size_t d, input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            ArrayBatch<typename input_field_t::value_type, cfg::stencil_size> stencil_values(args::batch_size);
            Batch<FluxValue<cfg>> flux_values(args::batch_size);
            auto context = flux_def.create_context();

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h = mesh.cell_length(level);

                for_each_interior_interface__same_level<run_type, get_type>(
                    mesh,
                    level,
                    flux_def.direction,
                    flux_def.stencil,
                    [&](auto& interface_cells, auto& comput_cells)
                    {
                        if constexpr (get_type == Get::Cells)
                        {
                            // times::timers_b.start("computation");
                            auto flux_values        = flux_function(comput_cells, field);
                            auto left_cell_contrib  = contribution(flux_values[0], h, h);
                            auto right_cell_contrib = contribution(flux_values[1], h, h);
                            // times::timers_b.stop("computation");
                            // times::timers_b.start("copy to field");
                            apply_contrib(interface_cells[0], left_cell_contrib);
                            apply_contrib(interface_cells[1], right_cell_contrib);
                            // times::timers_b.stop("copy to field");
                        }
                        else if constexpr (get_type == Get::CellBatches)
                        {
                            flux_values.resize(interface_cells.size());
                            times::timers_b.start("transform");
                            transform(comput_cells,
                                      stencil_values,
                                      [&](const auto& cell)
                                      {
                                          return field[cell];
                                      });
                            times::timers_b.stop("transform");

                            times::timers_b.start("computation");
                            flux_def.cons_flux_function__batch(comput_cells, flux_values, context, stencil_values);
                            auto factor = h_factor(h, h);
                            flux_values *= factor;
                            times::timers_b.stop("computation");
                            times::timers_b.start("copy to field");
                            apply_contrib(interface_cells[0], flux_values);
                            times::timers_b.stop("copy to field");
                            times::timers_b.start("computation");
                            flux_values *= -1;
                            times::timers_b.stop("computation");
                            times::timers_b.start("copy to field");
                            apply_contrib(interface_cells[1], flux_values);
                            times::timers_b.stop("copy to field");
                        }
                    });
            }

            // Level jumps (level -- level+1)
            for (std::size_t level = min_level; level < max_level; ++level)
            {
                auto h_l   = mesh.cell_length(level);
                auto h_lp1 = mesh.cell_length(level + 1);

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                {
                    for_each_interior_interface__level_jump_direction<run_type, get_type>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            if constexpr (get_type == Get::Cells)
                            {
                                // times::timers_b.start("computation");
                                auto flux_values        = flux_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(flux_values[0], h_lp1, h_l);
                                auto right_cell_contrib = contribution(flux_values[1], h_lp1, h_lp1);
                                // times::timers_b.stop("computation");
                                // times::timers_b.start("copy to field");
                                apply_contrib(interface_cells[0], left_cell_contrib);
                                apply_contrib(interface_cells[1], right_cell_contrib);
                                // times::timers_b.stop("copy to field");
                            }
                            else if constexpr (get_type == Get::CellBatches)
                            {
                                flux_values.resize(interface_cells.size());
                                // times::timers_b.start("transform");
                                transform(comput_cells,
                                          stencil_values,
                                          [&](const auto& cell)
                                          {
                                              return field[cell];
                                          });
                                // times::timers_b.stop("transform");

                                // times::timers_b.start("computation");
                                flux_def.cons_flux_function__batch(comput_cells, flux_values, context, stencil_values);
                                auto left_factor = h_factor(h_lp1, h_l);
                                flux_values *= left_factor;
                                // times::timers_b.stop("computation");
                                // times::timers_b.start("copy to field");
                                apply_contrib(interface_cells[0], flux_values);
                                // times::timers_b.stop("copy to field");
                                // times::timers_b.start("computation");
                                auto right_factor = h_factor(h_lp1, h_lp1);
                                flux_values *= -1 / left_factor * right_factor; // cancel left factor and apply right one
                                // times::timers_b.stop("computation");
                                // times::timers_b.start("copy to field");
                                apply_contrib(interface_cells[1], flux_values);
                                // times::timers_b.stop("copy to field");
                            }
                        });
                }
                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                {
                    for_each_interior_interface__level_jump_opposite_direction<run_type, get_type>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            if constexpr (get_type == Get::Cells)
                            {
                                // times::timers_b.start("computation");
                                auto flux_values        = flux_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(flux_values[0], h_lp1, h_lp1);
                                auto right_cell_contrib = contribution(flux_values[1], h_lp1, h_l);
                                // times::timers_b.stop("computation");
                                // times::timers_b.start("copy to field");
                                apply_contrib(interface_cells[0], left_cell_contrib);
                                apply_contrib(interface_cells[1], right_cell_contrib);
                                // times::timers_b.stop("copy to field");
                            }
                            else if constexpr (get_type == Get::CellBatches)
                            {
                                flux_values.resize(interface_cells.size());
                                // times::timers_b.start("transform");
                                transform(comput_cells,
                                          stencil_values,
                                          [&](const auto& cell)
                                          {
                                              return field[cell];
                                          });
                                // times::timers_b.stop("transform");

                                // times::timers_b.start("computation");
                                flux_def.cons_flux_function__batch(comput_cells, flux_values, context, stencil_values);
                                auto left_factor = h_factor(h_lp1, h_lp1);
                                flux_values *= left_factor;
                                // times::timers_b.stop("computation");
                                // times::timers_b.start("copy to field");
                                apply_contrib(interface_cells[0], flux_values);
                                // times::timers_b.stop("copy to field");
                                // times::timers_b.start("computation");
                                auto right_factor = h_factor(h_lp1, h_l);
                                flux_values *= -1 / left_factor * right_factor; // cancel left factor and apply right one
                                // times::timers_b.start("computation");
                                // times::timers_b.start("copy to field");
                                apply_contrib(interface_cells[1], flux_values);
                                // times::timers_b.stop("copy to field");
                            }
                        });
                }
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the boundary interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, class Func>
        void for_each_boundary_interface(std::size_t d, input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            for_each_level(mesh,
                           [&](auto level)
                           {
                               auto h = mesh.cell_length(level);

                               // Boundary in direction
                               for_each_boundary_interface__direction<run_type>(mesh,
                                                                                level,
                                                                                flux_def.direction,
                                                                                flux_def.stencil,
                                                                                [&](auto& cell, auto& comput_cells)
                                                                                {
                                                                                    auto flux_values  = flux_function(comput_cells, field);
                                                                                    auto cell_contrib = contribution(flux_values[0], h, h);
                                                                                    apply_contrib(cell, cell_contrib);
                                                                                });

                               // Boundary in opposite direction
                               for_each_boundary_interface__opposite_direction<run_type>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& cell, auto& comput_cells)
                                   {
                                       auto flux_values  = flux_function(comput_cells, field);
                                       auto cell_contrib = contribution(flux_values[1], h, h);
                                       apply_contrib(cell, cell_contrib);
                                   });
                           });
        }

        /**
         * This function is used in the Assembly class to iterate over the interior interfaces
         * and receive the Jacobian coefficients.
         */
        template <Run run_type = Run::Sequential, class Func>
        void for_each_interior_interface_and_coeffs(input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                auto jacobian_function = flux_def.jacobian_function ? flux_def.jacobian_function
                                                                    : flux_def.jacobian_function_as_conservative();

                if (!jacobian_function)
                {
                    std::cerr << "The jacobian function of operator '" << this->name() << "' has not been implemented." << std::endl;
                    std::cerr << "Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix." << std::endl;
                    exit(EXIT_FAILURE);
                }

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h = mesh.cell_length(level);

                    for_each_interior_interface__same_level<run_type>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            auto jacobians          = jacobian_function(comput_cells, field);
                            auto left_cell_contrib  = contribution(jacobians[0], h, h);
                            auto right_cell_contrib = contribution(jacobians[1], h, h);
                            apply_contrib(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                        });
                }

                // Level jumps (level -- level+1)
                for (std::size_t level = min_level; level < max_level; ++level)
                {
                    auto h_l   = mesh.cell_length(level);
                    auto h_lp1 = mesh.cell_length(level + 1);

                    //         |__|   l+1
                    //    |____|      l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface__level_jump_direction<run_type>(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto jacobians          = jacobian_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(jacobians[0], h_lp1, h_l);
                                auto right_cell_contrib = contribution(jacobians[1], h_lp1, h_lp1);
                                apply_contrib(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                    //    |__|        l+1
                    //       |____|   l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface__level_jump_opposite_direction<run_type>(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto jacobians          = jacobian_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(jacobians[0], h_lp1, h_lp1);
                                auto right_cell_contrib = contribution(jacobians[1], h_lp1, h_l);
                                apply_contrib(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                }
            }
        }

        /**
         * This function is used in the Assembly class to iterate over the boundary interfaces
         * and receive the Jacobian coefficients.
         */
        template <Run run_type = Run::Sequential, class Func>
        void for_each_boundary_interface_and_coeffs(input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                auto jacobian_function = flux_def.jacobian_function ? flux_def.jacobian_function
                                                                    : flux_def.jacobian_function_as_conservative();

                for_each_level(
                    mesh,
                    [&](auto level)
                    {
                        auto h = mesh.cell_length(level);

                        // Boundary in direction
                        for_each_boundary_interface__direction<run_type>(mesh,
                                                                         level,
                                                                         flux_def.direction,
                                                                         flux_def.stencil,
                                                                         [&](auto& cell, auto& comput_cells)
                                                                         {
                                                                             auto jacobians    = jacobian_function(comput_cells, field);
                                                                             auto cell_contrib = contribution(jacobians[0], h, h);
                                                                             apply_contrib(cell, comput_cells, cell_contrib);
                                                                         });

                        // Boundary in opposite direction
                        for_each_boundary_interface__opposite_direction<run_type>(mesh,
                                                                                  level,
                                                                                  flux_def.direction,
                                                                                  flux_def.stencil,
                                                                                  [&](auto& cell, auto& comput_cells)
                                                                                  {
                                                                                      auto jacobians = jacobian_function(comput_cells, field);
                                                                                      auto cell_contrib = contribution(jacobians[1], h, h);
                                                                                      apply_contrib(cell, comput_cells, cell_contrib);
                                                                                  });
                    });
            }
        }
    };

} // end namespace samurai
