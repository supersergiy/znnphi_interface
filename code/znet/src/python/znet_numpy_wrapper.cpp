#include <znet.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>

namespace py = pybind11;

class ZnetNumpyWrapper {
   private:
      znn::phi::Znet *zn;
   public:
      ZnetNumpyWrapper(const std::string &weights_path) 
      {
         zn = new znn::phi::Znet(weights_path);
      }

      std::vector<size_t> get_in_shape() 
      {
         return zn->in_shape;
      }

      std::vector<size_t> get_out_shape() 
      {
         return zn->out_shape;
      }

      py::array_t<float> forward(py::array_t<float, py::array::c_style | py::array::forcecast> in)
      {
         py::buffer_info in_info = in.request();
         auto in_ptr = static_cast<float *>(in_info.ptr);

         if (in_info.size != zn->input_size) {
            std::cerr << "Erroneus input size." << std::endl;
            std::cerr << "Expected " << zn->input_size << " elements, but got " << in_info.size << std::endl;
            exit(EXIT_FAILURE);
         }
         std::cerr << "Copying " << zn->input_size*sizeof(float) << " bytes into the user input. " << std::endl;
         std::memcpy(zn->tensors["user_input"]->data(), in_ptr, zn->input_size*sizeof(float));

         zn->forward();
         std::cout << "Forward Finished\n";
         auto out_data = zn->tensors["user_output"]->data();

         return py::array(py::buffer_info(out_data, sizeof(float),
                                          py::format_descriptor<float>::format(),
                                          zn->out_dim, zn->out_shape, zn->out_strides));
      }

};

namespace py = pybind11;

PYBIND11_MODULE(znet, m) {
    py::class_<ZnetNumpyWrapper>(m, "znet")
        .def(py::init<const std::string &>(), py::arg("weights_path")) 
        .def("forward", &ZnetNumpyWrapper::forward)
        .def("get_in_shape", &ZnetNumpyWrapper::get_in_shape)
        .def("get_out_shape", &ZnetNumpyWrapper::get_out_shape);
}
