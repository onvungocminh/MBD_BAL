#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>

#include "MBD_distance_2d.h"

namespace details
{
    namespace py = pybind11;
    inline const bool check_dim(const py::array_t<std::uint8_t>& img, const py::array_t<std::uint8_t>& seed, const py::array_t<std::uint8_t>& dest) noexcept
    {
        return img.ndim() == 2 && seed.ndim() == img.ndim() && dest.ndim() == img.ndim() && img.shape(0) == seed.shape(0) && img.shape(0) == dest.shape(0) && img.shape(1) == seed.shape(1) && img.shape(1) == dest.shape(1);
    }
}

PYBIND11_MODULE(MBD, m)
{
    namespace py = pybind11;
    m.def("MBD_cut", [](const py::array_t<std::uint8_t>& img, const py::array_t<std::uint8_t>& seed, const py::array_t<std::uint8_t>& dest)
    {
        if (!details::check_dim(img, seed, dest))
            throw std::invalid_argument("Invalid input arrays dimensions. Sould be 2D arrays with the same shape");
        py::array_t<std::uint8_t> out({img.shape(0), img.shape(1)});
        py::buffer_info img_buf = img.request(), seed_buf = seed.request(), dest_buf = dest.request(), out_buf = out.request();
        MBD_cut(static_cast<const unsigned char*>(img_buf.ptr), static_cast<const unsigned char*>(seed_buf.ptr), static_cast<const unsigned char*>(dest_buf.ptr), static_cast<unsigned char*>(out_buf.ptr), img.shape(0), img.shape(1));
        return out;
    }, "Compute a 2D MBD cut");
}