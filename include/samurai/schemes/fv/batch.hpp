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

        std::array<dynamic_vector_t, array_size> m_batch;

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
        }

        inline auto& operator[](std::size_t index_in_array)
        {
            return m_batch[index_in_array];
        }

        inline const auto& operator[](std::size_t index_in_array) const
        {
            return m_batch[index_in_array];
        }

        inline void reserve(std::size_t batch_size)
        {
            if constexpr (array_size == 1)
            {
                m_batch.reserve(batch_size);
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    m_batch[i].reserve(batch_size);
                }
            }
        }

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
                    m_batch[i].resize(batch_size);
                }
            }
        }

        // inline void add(const CollapsStdArray<T, array_size>& values)
        inline void add(const std::array<T, array_size>& values)
        {
            if constexpr (array_size == 1)
            {
                m_batch.push_back(values);
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    m_batch[i].push_back(values[i]);
                }
            }
        }

        inline void clear()
        {
            if constexpr (array_size == 1)
            {
                m_batch.clear();
            }
            else
            {
                for (std::size_t i = 0; i < array_size; ++i)
                {
                    m_batch[i].clear();
                }
            }
        }

        inline bool empty() const
        {
            if constexpr (array_size == 1)
            {
                return m_batch[0].empty();
            }
            else
            {
                return m_batch[0].empty();
            }
        }
    };

    template <class T>
    using Batch = StdVectorWrapper<T>;

    // template <class T>
    // class ArrayBatch<T, 1>

    // template <class T>
    // class Batch
    // {
    //   public:

    //     using value_type = T;

    //   private:

    //     using dynamic_vector_t = StdVectorWrapper<T>;

    //     dynamic_vector_t m_batch;

    //   public:

    //     Batch()
    //     {
    //     }

    //     Batch(std::size_t batch_size)
    //     {
    //         resize(batch_size);
    //     }

    //     inline auto& batch()
    //     {
    //         return m_batch;
    //     }

    //     inline const auto& batch() const
    //     {
    //         return m_batch;
    //     }

    //     inline std::size_t size() const
    //     {
    //         return m_batch.size();
    //     }

    //     inline void reserve(std::size_t batch_size)
    //     {
    //         m_batch.reserve(batch_size);
    //     }

    //     inline void resize(std::size_t batch_size)
    //     {
    //         m_batch.resize(batch_size);
    //     }

    //     inline void add(const T& value)
    //     {
    //         m_batch.push_back(value);
    //     }

    //     inline void clear()
    //     {
    //         if constexpr (array_size == 1)
    //         {
    //             m_batch.clear();
    //         }
    //         else
    //         {
    //             for (std::size_t i = 0; i < array_size; ++i)
    //             {
    //                 m_batch[i].clear();
    //             }
    //         }
    //     }

    //     inline bool empty() const
    //     {
    //         return m_batch.empty();
    //     }
    // };

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
