#pragma once

#include <thrust/device_vector.h>
#include "util.h"

namespace thrust {

  template<typename base_iterator>
  struct multi_device_iterator {
  };

  template<typename T>
  struct multi_device_vector {
    typedef multi_device_vector<T> self_t;
    typedef thrust::device_vector<T> base_vector_t;
    typedef std::vector<base_vector_t *> vector_t;
    //typedef thrust::device_vector<pointer_type> vector_storage_t;
    typedef typename base_vector_t::iterator base_iterator;
    typedef multi_device_iterator<base_iterator> iterator;

    multi_device_vector() {}

    explicit multi_device_vector(int size, const T& prefix) {
      cudaGetDeviceCount(&m_tiles);
      m_tile_size = util::__ceil(size, m_tiles);

      int allocated = 0;
      for(int i = 0; i < m_tiles - 1; ++i) {
        cudaSetDevice(i);
        base_vector_t *p = new base_vector_t(m_tile_size, prefix);
        m_data.push_back(p);
        allocated += m_tile_size;
      }

      cudaSetDevice(m_tiles - 1);
      // Allocate final chunk
      base_vector_t *p = new base_vector_t(size - allocated, prefix);
      m_data.push_back(p);
    }

    template<typename _iterator>
    multi_device_vector(_iterator begin, _iterator end) {
      cudaGetDeviceCount(&m_tiles);
      const int size = end-begin;
      m_tile_size = util::__ceil(size, m_tiles);

      int allocated = 0;
      for(int i = 0; i < m_tiles - 1; ++i) {
        cudaSetDevice(i);
        base_vector_t *p = new base_vector_t(begin + allocated,
                           begin + allocated + m_tile_size);
        m_data.push_back(p);
        allocated += m_tile_size;
      }

      cudaSetDevice(m_tiles - 1);
      // Allocate final chunk
      base_vector_t *p = new base_vector_t(begin + allocated, begin + size);
      m_data.push_back(p);
    }

    template<typename S>
    multi_device_vector(const S& s) : self_t(s.begin(), s.end()) {}

    ~multi_device_vector() {
      for(int i = 0; i < m_tiles; ++i)
        delete m_data[i];
    }

  private:
    int m_tiles;
    int m_tile_size;
    vector_t m_data;
    //vector_storage_t m_pointers;
  };

} // namespace thrust
