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
        using size_type = typename base_class::size_type;

        using typename base_class::field_value_type;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;

        using cell_t = typename mesh_t::cell_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

        struct BatchMemory
        {
            ArrayBatch<cell_t, 2> interface_batch;
            ArrayBatch<cell_t, cfg::stencil_size> comput_stencil_batch;
            ArrayBatch<typename input_field_t::value_type, cfg::stencil_size> stencil_values;
            Batch<FluxValue<cfg>> flux_values;
            BatchData batch_data;

            void resize(std::size_t size)
            {
                interface_batch.resize(size);
                comput_stencil_batch.resize(size);
                stencil_values.resize(size);
                flux_values.resize(size);
            }
        };

      private:

        FluxDefinition<cfg> m_flux_definition;
        BatchMemory m_batch_memory;

        bool m_include_boundary_fluxes = true;
        bool m_enable_batches          = true;

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

        void enable_batches(bool enable)
        {
            m_enable_batches = enable;
        }

        bool enable_batches() const
        {
            return m_enable_batches;
        }

      private:

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

      public:

        inline field_value_type flux_value_cmpnent(const FluxValue<cfg>& flux_value, [[maybe_unused]] std::size_t field_i) const
        {
            if constexpr (output_field_size == 1)
            {
                return flux_value;
            }
            else
            {
                return flux_value(static_cast<flux_index_type>(field_i));
            }
        }

      private:

        template <class Func>
        inline void
        call_flux_function__batch(const NormalFluxDefinition<cfg>& flux_def, double left_factor, double right_factor, Func&& apply_contrib)
        {
            auto& interface_batch      = m_batch_memory.interface_batch;
            auto& comput_stencil_batch = m_batch_memory.comput_stencil_batch;
            auto& stencil_values       = m_batch_memory.stencil_values;
            auto& flux_values          = m_batch_memory.flux_values;
            auto& batch_data           = m_batch_memory.batch_data;

            batch_data.size = interface_batch.position();
            flux_values.resize(batch_data.size);

            flux_def.cons_flux_function__batch(batch_data, comput_stencil_batch, flux_values, stencil_values);

            flux_values *= left_factor;
            apply_contrib(interface_batch[0], flux_values);
            flux_values *= -1. / left_factor * right_factor; // add minus sign, cancel left factor and apply right one
            apply_contrib(interface_batch[1], flux_values);

            stencil_values.reset_position();
            interface_batch.reset_position();
            comput_stencil_batch.reset_position();
        }

        template <class Func>
        inline void call_flux_function_boundary__batch(const NormalFluxDefinition<cfg>& flux_def, double factor, Func&& apply_contrib)
        {
            auto& interface_batch      = m_batch_memory.interface_batch;
            auto& comput_stencil_batch = m_batch_memory.comput_stencil_batch;
            auto& stencil_values       = m_batch_memory.stencil_values;
            auto& flux_values          = m_batch_memory.flux_values;
            auto& batch_data           = m_batch_memory.batch_data;

            batch_data.size = interface_batch.position();
            flux_values.resize(batch_data.size);

            flux_def.cons_flux_function__batch(batch_data, comput_stencil_batch, flux_values, stencil_values);

            flux_values *= factor;
            apply_contrib(interface_batch[0], flux_values);

            stencil_values.reset_position();
            interface_batch.reset_position();
            comput_stencil_batch.reset_position();
        }

      public:

        /**
         * This function is used in the Explicit class to iterate over the interior interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_interior_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            auto& interface_batch      = m_batch_memory.interface_batch;
            auto& comput_stencil_batch = m_batch_memory.comput_stencil_batch;
            auto& stencil_values       = m_batch_memory.stencil_values;
            auto& batch_data           = m_batch_memory.batch_data;
            if constexpr (get_type == Get::CellBatches)
            {
                m_batch_memory.resize(args::batch_size);
                if (flux_def.create_temp_variables)
                {
                    batch_data.temp_variables = flux_def.create_temp_variables();
                }
            }

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h = mesh.cell_length(level);

                batch_data.cell_length = h;
                auto factor            = h_factor(h, h);

                for_each_interior_interface__same_level<run_type, Get::Intervals>(
                    mesh,
                    level,
                    flux_def.direction,
                    flux_def.stencil,
                    [&](auto& interface_it, auto& comput_stencil_it)
                    {
                        if constexpr (get_type == Get::Cells)
                        {
                            for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                            {
                                auto flux_values        = flux_function(comput_stencil_it.cells(), field);
                                auto left_cell_contrib  = contribution(flux_values[0], h, h);
                                auto right_cell_contrib = contribution(flux_values[1], h, h);
                                apply_contrib(interface_it.cells()[0], left_cell_contrib);
                                apply_contrib(interface_it.cells()[1], right_cell_contrib);

                                interface_it.move_next();
                                comput_stencil_it.move_next();
                            }
                        }
                        else if constexpr (get_type == Get::CellBatches)
                        {
                            std::size_t to_process = comput_stencil_it.interval().size();
                            while (to_process > 0)
                            {
                                auto n = std::min(to_process, args::batch_size - interface_batch.position());

                                // Copy field values
                                comput_stencil_it.copy_values_to_batch(n, stencil_values, field);

                                interface_it.copy_to_batch(n, interface_batch);
                                comput_stencil_it.copy_to_batch(n, comput_stencil_batch);

                                to_process -= n;
                                if (interface_batch.position() == args::batch_size)
                                {
                                    call_flux_function__batch(flux_def, factor, factor, std::forward<Func>(apply_contrib));
                                }
                            }
                        }
                    });

                if constexpr (get_type == Get::CellBatches)
                {
                    if (interface_batch.position() > 0)
                    {
                        call_flux_function__batch(flux_def, factor, factor, std::forward<Func>(apply_contrib));
                    }
                }
            }

            // Level jumps (level -- level+1)
            for (std::size_t level = min_level; level < max_level; ++level)
            {
                auto h_l   = mesh.cell_length(level);
                auto h_lp1 = mesh.cell_length(level + 1);

                batch_data.cell_length = h_lp1;

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                {
                    auto left_factor  = h_factor(h_lp1, h_l);
                    auto right_factor = h_factor(h_lp1, h_lp1);

                    for_each_interior_interface__level_jump_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            if constexpr (get_type == Get::Cells)
                            {
                                for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                {
                                    auto flux_values        = flux_function(comput_stencil_it.cells(), field);
                                    auto left_cell_contrib  = contribution(flux_values[0], h_lp1, h_l);
                                    auto right_cell_contrib = contribution(flux_values[1], h_lp1, h_lp1);
                                    apply_contrib(interface_it.cells()[0], left_cell_contrib);
                                    apply_contrib(interface_it.cells()[1], right_cell_contrib);

                                    interface_it.move_next();
                                    comput_stencil_it.move_next();
                                }
                            }
                            else if constexpr (get_type == Get::CellBatches)
                            {
                                std::size_t to_process = comput_stencil_it.interval().size();
                                while (to_process > 0)
                                {
                                    auto n = std::min(to_process, args::batch_size - interface_batch.position());

                                    // Copy field values
                                    comput_stencil_it.copy_values_to_batch(n, stencil_values, field);

                                    interface_it.copy_to_batch(n, interface_batch);
                                    comput_stencil_it.copy_to_batch(n, comput_stencil_batch);

                                    to_process -= n;
                                    if (interface_batch.position() == args::batch_size)
                                    {
                                        call_flux_function__batch(flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                                    }
                                }
                            }
                        });
                    if constexpr (get_type == Get::CellBatches)
                    {
                        if (interface_batch.position() > 0)
                        {
                            call_flux_function__batch(flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                        }
                    }
                }
                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                {
                    auto left_factor  = h_factor(h_lp1, h_lp1);
                    auto right_factor = h_factor(h_lp1, h_l);

                    for_each_interior_interface__level_jump_opposite_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            if constexpr (get_type == Get::Cells)
                            {
                                for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                {
                                    auto flux_values        = flux_function(comput_stencil_it.cells(), field);
                                    auto left_cell_contrib  = contribution(flux_values[0], h_lp1, h_lp1);
                                    auto right_cell_contrib = contribution(flux_values[1], h_lp1, h_l);
                                    apply_contrib(interface_it.cells()[0], left_cell_contrib);
                                    apply_contrib(interface_it.cells()[1], right_cell_contrib);

                                    interface_it.move_next();
                                    comput_stencil_it.move_next();
                                }
                            }
                            else if constexpr (get_type == Get::CellBatches)
                            {
                                std::size_t to_process = comput_stencil_it.interval().size();
                                while (to_process > 0)
                                {
                                    auto n = std::min(to_process, args::batch_size - interface_batch.position());

                                    // Copy field values
                                    comput_stencil_it.copy_values_to_batch(n, stencil_values, field);

                                    interface_it.copy_to_batch(n, interface_batch);
                                    comput_stencil_it.copy_to_batch(n, comput_stencil_batch);

                                    to_process -= n;
                                    if (interface_batch.position() == args::batch_size)
                                    {
                                        call_flux_function__batch(flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                                    }
                                }
                            }
                        });
                    if constexpr (get_type == Get::CellBatches)
                    {
                        if (interface_batch.position() > 0)
                        {
                            call_flux_function__batch(flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                        }
                    }
                }
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the boundary interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_boundary_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            auto& interface_batch      = m_batch_memory.interface_batch;
            auto& comput_stencil_batch = m_batch_memory.comput_stencil_batch;
            auto& stencil_values       = m_batch_memory.stencil_values;
            auto& batch_data           = m_batch_memory.batch_data;
            if constexpr (get_type == Get::CellBatches)
            {
                m_batch_memory.resize(args::batch_size);
                if (flux_def.create_temp_variables)
                {
                    batch_data.temp_variables = flux_def.create_temp_variables();
                }
            }

            for_each_level(mesh,
                           [&](auto level)
                           {
                               auto h = mesh.cell_length(level);

                               batch_data.cell_length = h;

                               auto factor = h_factor(h, h);

                               // Boundary in direction
                               for_each_boundary_interface__direction<run_type, Get::Intervals>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& interface_it, auto& comput_stencil_it)
                                   {
                                       if constexpr (get_type == Get::Cells)
                                       {
                                           for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                           {
                                               auto flux_values  = flux_function(comput_stencil_it.cells(), field);
                                               auto cell_contrib = contribution(flux_values[0], h, h);
                                               apply_contrib(interface_it.cells()[0], cell_contrib);

                                               interface_it.move_next();
                                               comput_stencil_it.move_next();
                                           }
                                       }
                                       else if constexpr (get_type == Get::CellBatches)
                                       {
                                           std::size_t to_process = comput_stencil_it.interval().size();
                                           while (to_process > 0)
                                           {
                                               auto n = std::min(to_process, args::batch_size - interface_batch.position());

                                               // Copy field values
                                               comput_stencil_it.copy_values_to_batch(n, stencil_values, field);

                                               interface_it.copy_to_batch(n, interface_batch);
                                               comput_stencil_it.copy_to_batch(n, comput_stencil_batch);

                                               to_process -= n;
                                               if (interface_batch.position() == args::batch_size)
                                               {
                                                   call_flux_function_boundary__batch(flux_def, factor, std::forward<Func>(apply_contrib));
                                               }
                                           }
                                       }
                                       //    else if constexpr (get_type == Get::CellBatches)
                                       //    {
                                       //        batch_data.size = comput_cells.size();
                                       //        flux_values.resize(batch_data.size);
                                       //        // times::timers_b.start("transform");
                                       //        transform(comput_cells,
                                       //                  stencil_values,
                                       //                  [&](const auto& c)
                                       //                  {
                                       //                      return field[c];
                                       //                  });

                                       //        // times::timers_b.stop("transform");

                                       //        // times::timers_b.start("computation");
                                       //        flux_def.cons_flux_function__batch(batch_data, comput_cells, flux_values, stencil_values);
                                       //        auto factor = h_factor(h, h);
                                       //        flux_values *= factor;
                                       //        // times::timers_b.stop("computation");
                                       //        // times::timers_b.start("copy to field");

                                       //        // static_assert(std::is_same_v<void, decltype(cell)>);
                                       //        apply_contrib(cell, flux_values);
                                       //        // times::timers_b.stop("copy to field");
                                       //    }
                                   });

                               if constexpr (get_type == Get::CellBatches)
                               {
                                   if (interface_batch.position() > 0)
                                   {
                                       call_flux_function_boundary__batch(flux_def, factor, std::forward<Func>(apply_contrib));
                                   }
                               }

                               // Boundary in opposite direction
                               for_each_boundary_interface__opposite_direction<run_type, Get::Intervals>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& interface_it, auto& comput_stencil_it)
                                   {
                                       if constexpr (get_type == Get::Cells)
                                       {
                                           for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                           {
                                               auto flux_values  = flux_function(comput_stencil_it.cells(), field);
                                               auto cell_contrib = contribution(flux_values[1], h, h);
                                               apply_contrib(interface_it.cells()[0], cell_contrib);

                                               interface_it.move_next();
                                               comput_stencil_it.move_next();
                                           }
                                       }
                                       else if constexpr (get_type == Get::CellBatches)
                                       {
                                           std::size_t to_process = comput_stencil_it.interval().size();
                                           while (to_process > 0)
                                           {
                                               auto n = std::min(to_process, args::batch_size - interface_batch.position());

                                               // Copy field values
                                               comput_stencil_it.copy_values_to_batch(n, stencil_values, field);

                                               interface_it.copy_to_batch(n, interface_batch);
                                               comput_stencil_it.copy_to_batch(n, comput_stencil_batch);

                                               to_process -= n;
                                               if (interface_batch.position() == args::batch_size)
                                               {
                                                   call_flux_function_boundary__batch(flux_def, -factor, std::forward<Func>(apply_contrib));
                                               }
                                           }
                                       }
                                       //    else if constexpr (get_type == Get::CellBatches)
                                       //    {
                                       //        batch_data.size = comput_cells.size();
                                       //        flux_values.resize(batch_data.size);
                                       //        // times::timers_b.start("transform");
                                       //        transform(comput_cells,
                                       //                  stencil_values,
                                       //                  [&](const auto& c)
                                       //                  {
                                       //                      return field[c];
                                       //                  });

                                       //        // times::timers_b.stop("transform");

                                       //        // times::timers_b.start("computation");
                                       //        flux_def.cons_flux_function__batch(batch_data, comput_cells, flux_values, stencil_values);
                                       //        auto factor = h_factor(h, h);
                                       //        flux_values *= -factor;
                                       //        // times::timers_b.stop("computation");
                                       //        // times::timers_b.start("copy to field");
                                       //        apply_contrib(cell, flux_values);
                                       //        // times::timers_b.stop("copy to field");
                                       //    }
                                   });

                               if constexpr (get_type == Get::CellBatches)
                               {
                                   if (interface_batch.position() > 0)
                                   {
                                       call_flux_function_boundary__batch(flux_def, -factor, std::forward<Func>(apply_contrib));
                                   }
                               }
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
