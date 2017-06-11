#include "znet.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>

namespace py = pybind11;

class ZnetNumpyWrapper {
   private:
      znn::phi::Znet zn;
   public:
      ZnetNumpyWrapper(const std::string &weights_path) 
      {
         zn = znn::phi::Znet(weights_path);
      }

      py::array_t<float> forward(py::array_t<float, py::array::c_style | py::array::forcecast> in)
      {
         py::buffer_info in_info = in.request();
         auto in_ptr = static_cast<float *>(in_info.ptr);

         std::memcpy(zn.tensors["user_input"]->data(), in_ptr, zn.input_size);
         zn.forward();
			auto out_data = zn.tensors["user_output"]->data();

         std::cout << zn.out_dim << std::endl;
         std::cout << zn.out_shape.size() << std::endl;
         std::cout << zn.out_strides.size() << std::endl;

		   return py::array(py::buffer_info(out_data, sizeof(float),
                                          py::format_descriptor<float>::format(),
                                          zn.out_dim, zn.out_shape, zn.out_strides));
      }
};

namespace py = pybind11;

PYBIND11_MODULE(znet, m) {
    py::class_<ZnetNumpyWrapper>(m, "znet")
        .def(py::init<const std::string &>(), py::arg("weights_path") = "./weights/") 
        .def("forward", &ZnetNumpyWrapper::forward);
}
