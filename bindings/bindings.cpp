#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // to enable conversion between std::vector and list

#include <string>
#include <vector>

#include "Neural.h"

namespace py = pybind11;

PYBIND11_MODULE(neuralnet, m, py::mod_gil_not_used()) {
  py::class_<NeuralNetwork>(m, "NeuralNetwork")
      .def(py::init<const std::string &>())
      .def("guess", [](NeuralNetwork &self, const std::vector<float> &input) {
        std::vector<float> out = self.forward(input);
        return out;
      });
}
