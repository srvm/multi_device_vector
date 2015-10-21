#pragma once

#include <thrust/device_vector.h>
#include "iterator.h"

namespace thrust {

  template<typename T>
  struct multi_device_vector {
    typedef multi_device_vector<T> self_t;

    typedef thrust::device_vector<T> base_vector_t;
    typedef typename base_vector_t::iterator base_iterator;
    typedef multi_device_iterator<base_iterator> iterator;
    typedef multi_device_iterator<base_iterator> const_iterator;

    typedef thrust::device_ptr<T> pointer_t;
    typedef thrust::device_reference<T> reference_t;
    typedef thrust::device_vector<base_iterator> storage_vector_t;
    typedef std::vector<base_vector_t *> vector_t;

    multi_device_vector() : m_data{}, m_pointers{} {}

    explicit multi_device_vector(int size, const T& prefix) :
             m_size(size) {
      cudaGetDeviceCount(&m_tiles);
      m_tile_size = (size - 1)/m_tiles + 1;

      int original_device = 0;
      cudaGetDevice(&original_device);

      int allocated = 0;
      for(int i = 0; i < m_tiles - 1; ++i) {
        cudaSetDevice(i);
        base_vector_t *p = new base_vector_t(m_tile_size, prefix);
        m_data.push_back(p);
        m_pointers.push_back(p->begin());
        allocated += m_tile_size;
      }

      cudaSetDevice(m_tiles - 1);
      // Allocate final chunk
      base_vector_t *p = new base_vector_t(size - allocated, prefix);
      m_data.push_back(p);
      m_pointers.push_back(p->begin());

      cudaSetDevice(original_device);
      m_pointers_start = m_pointers.data().get();
    }

    template<typename _iterator>
    multi_device_vector(_iterator begin, _iterator end) :
                        m_size(end-begin) {
      cudaGetDeviceCount(&m_tiles);
      const int size = end-begin;
      m_tile_size = (size - 1)/m_tiles + 1;

      int original_device = 0;
      cudaGetDevice(&original_device);

      int allocated = 0;
      for(int i = 0; i < m_tiles - 1; ++i) {
        cudaSetDevice(i);
        base_vector_t *p = new base_vector_t(begin + allocated,
                           begin + allocated + m_tile_size);
        m_data.push_back(p);
        m_pointers.push_back(p->begin());
        allocated += m_tile_size;
      }

      cudaSetDevice(m_tiles - 1);
      // Allocate final chunk
      base_vector_t *p = new base_vector_t(begin + allocated, begin + size);
      m_data.push_back(p);
      m_pointers.push_back(p->begin());

      cudaSetDevice(original_device);
      m_pointers_start = m_pointers.data().get();
    }

    template<typename S>
    multi_device_vector(const S& s) : self_t(s.begin(), s.end()) {}

    ~multi_device_vector() {
      for(int i = 0; i < m_tiles; ++i)
        delete m_data[i];
    }

    __host__ __device__
    iterator begin()
    { return iterator(m_pointers_start, m_tile_size); }

    __host__ __device__
    const_iterator begin() const
    { return const_iterator(m_pointers_start, m_tile_size); }

    __host__ __device__
    iterator end()
    { return begin() + m_size; }

    __host__ __device__
    const_iterator end() const
    { return begin() + m_size; }

    __device__
    T operator[](int index) const {
      return *(begin() + index);
    }

    __device__
    reference_t operator[](int index) {
      return *(begin() + index);
    }

    __host__ __device__
    int size() const
    { return m_size; }

  private:
    int m_tiles;
    int m_tile_size;
    int m_size;
    vector_t m_data;
    storage_vector_t m_pointers;
    base_iterator *m_pointers_start;
  };

} // namespace thrust
