#pragma once
#include "std_vector_wrapper.hpp"

namespace samurai
{
    template <class T, std::size_t array_size>
    class ArrayBatch
    {
      public:

        using value_type = T;

      private:

        using dynamic_vector_t = StdVectorWrapper<T>;
        // using dynamic_vector_t = xt::xtensor<T, 1>;

        std::array<dynamic_vector_t, array_size> m_batch;

        std::size_t m_add_counter = 0;

      public:

        ArrayBatch()
        {
        }

        ArrayBatch(std::size_t batch_size)
        {
            resize(batch_size);
        }

        inline auto& batch()
        {
            return m_batch;
        }

        inline const auto& batch() const
        {
            return m_batch;
        }

        inline std::size_t size() const
        {
            return m_batch[0].size();
            // return m_size;
        }

        inline const auto& add_counter() const
        {
            return m_add_counter;
        }

        inline auto& add_counter()
        {
            return m_add_counter;
        }

        inline void reset_add_counter()
        {
            m_add_counter = 0;
        }

        inline auto& operator[](std::size_t index_in_array)
        {
            return m_batch[index_in_array];
        }

        inline const auto& operator[](std::size_t index_in_array) const
        {
            return m_batch[index_in_array];
        }

        // inline void reserve(std::size_t batch_size)
        // {
        //     if constexpr (array_size == 1)
        //     {
        //         m_batch.reserve(batch_size);
        //     }
        //     else
        //     {
        //         for (std::size_t i = 0; i < array_size; ++i)
        //         {
        //             if constexpr (std::is_same_v<dynamic_vector_t, xt::xtensor<T, 1>>)
        //             {
        //                 m_batch[i].data().reserve(batch_size);
        //             }
        //             else
        //             {
        //                 m_batch[i].reserve(batch_size);
        //             }
        //         }
        //     }
        // }

        inline void resize(std::size_t batch_size)
        {
            if constexpr (array_size == 1)
            {
                m_batch.resize(batch_size);
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    if constexpr (std::is_same_v<dynamic_vector_t, xt::xtensor<T, 1>>)
                    {
                        m_batch[i].resize({batch_size});
                    }
                    else
                    {
                        m_batch[i].resize(batch_size);
                    }
                }
            }
        }

        // inline void add(const CollapsStdArray<T, array_size>& values)
        inline void add(const std::array<T, array_size>& values)
        {
            if constexpr (array_size == 1)
            {
                m_batch[m_add_counter].values;
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    m_batch[i][m_add_counter] = values[i];
                }
            }
            m_add_counter++;
        }

        template <class Func>
        inline void add(const std::array<T, array_size>& values, Func&& copy)
        {
            if constexpr (array_size == 1)
            {
                m_batch[m_add_counter].values;
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    copy(m_batch[i][m_add_counter], values[i]);
                }
            }
            m_add_counter++;
        }

        // inline bool empty() const
        // {
        //     if constexpr (array_size == 1)
        //     {
        //         return m_batch[0].empty();
        //     }
        //     else
        //     {
        //         return m_batch[0].empty();
        //     }
        // }
    };

    template <class T>
    using Batch = StdVectorWrapper<T>;

    template <class T1, class T2, std::size_t size, class Func>
    void transform(const ArrayBatch<T1, size>& input, ArrayBatch<T2, size>& output, Func&& op)
    {
        output.resize(input.size());
        for (std::size_t i = 0; i < size; ++i)
        {
            for (std::size_t j = 0; j < input.size(); ++j)
            {
                output[i][j] = op(input[i][j]);
            }
        }
    }

} // end namespace samurai
