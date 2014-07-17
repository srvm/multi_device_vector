#pragma once

namespace thrust {

  template<typename base_iterator>
  struct multi_device_iterator {
    typedef base_iterator iterator;
    typedef multi_device_iterator<iterator> self_t;

    typedef typename std::iterator_traits<iterator>::difference_type difference_type;
    typedef typename std::iterator_traits<iterator>::value_type value_type;
    typedef typename std::iterator_traits<iterator>::reference reference;
    typedef typename std::iterator_traits<iterator>::pointer pointer;
    typedef typename std::iterator_traits<iterator>::iterator_category iterator_category;

    typedef value_type T;
    typedef thrust::device_ptr<T> pointer_t;
    typedef pointer_t *storage_t;

    __host__ __device__
    multi_device_iterator(storage_t v,
                          int tile_size,
                          int offset = 0):
      m_v(v), m_tile_size(tile_size), m_pointer(offset) {}

    __device__
    reference operator*() {
      T *v_start = thrust::raw_pointer_cast(*(m_v + (m_pointer / m_tile_size)));
      return thrust::device_reference<T>(thrust::device_pointer_cast(
            v_start + (m_pointer % m_tile_size)));
    }

    // prefix
    __host__ __device__
    self_t& operator++()
    { m_pointer++; return *this; }

    // postfix
    __host__ __device__
    self_t operator++(int)
    { self_t r(*this); ++(*this); return r; }

    // prefix
    __host__ __device__
    self_t& operator--()
    { m_pointer--; return *this; }

    // postfix
    __host__ __device__
    self_t operator--(int)
    { self_t r(*this); --(*this); return r; }

    __host__ __device__
    bool operator==(const self_t& rhs) const
    { return m_pointer == rhs.m_pointer; }

    __host__ __device__
    bool operator!=(const self_t& rhs) const
    { return m_pointer != rhs.m_pointer; }

    __host__ __device__
    self_t operator+(difference_type i) const
    { return self_t(m_v, m_pointer + i); }

    __host__ __device__
    self_t& operator+=(difference_type i)
    { m_pointer += i; return *this; }

    __host__ __device__
    self_t& operator-=(difference_type i)
    { m_pointer -= i; return *this; }

    __host__ __device__
    self_t operator-(difference_type i) const
    { return self_t(m_v, m_pointer - i); }

    __host__ __device__
    difference_type operator-(self_t s) const
    { return (m_pointer - s.m_pointer); }

  private:
    storage_t m_v;
    int m_pointer;
    int m_tile_size;
  };

} // namespace thrust
