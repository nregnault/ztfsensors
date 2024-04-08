// #include <assert>
#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


typedef py::array_t<double> npa;


// the pocket model hosts 
class _PocketModel {
public:
    _PocketModel(double alpha, double cmax, double beta, double nmax)
        : _alpha(alpha), _cmax(cmax), _beta(beta), _nmax(nmax)
    {
    }

    ~_PocketModel() { }

    // transfer of electrons from the pocket to the pixels.
    // We expect it to be an increasing function of the pocket contents.
    // We model it as a power function of the pocket charge.
    double _flush(double q_i) const
    {
      if(q_i <= 0.)
          {
              //	  assert(q_i >= 0.);
              q_i = 0.;
              return 0.;
          }
      double x = q_i / _cmax;
      return _cmax * pow(x, _alpha);
    }

    // transfer of electrons from the pixel to the pocket.
    // It is a decreasing function of the pocket content.
    // and an increasing function of the pixel charge.
    // We model this as the product of two power functions,
    // since we know nothing of the physics at play.
    double _fill(double q_i, double n_i) const
    {
        double x = q_i / _cmax;
        double y = n_i / _nmax;
        if(y<0.)
            return 0.;
        return _cmax * pow(1.-x, _alpha) * pow(y, _beta);
    }

    // apply the model to a line
    // (i.e. simulate the effect on the contents of the serial register,
    // as the pixels are read out).
    npa   apply(const npa & pix)
    {
        std::vector<size_t> x_shape(pix.shape(), pix.shape()+pix.ndim());
        py::array_t<double> res(x_shape);

        // in practice, this should almost never happen
        auto size = pix.size();
        if(_pocket.size() != size)
            _pocket.resize(pix.size()+1);
        _pocket[0] = 0.;

        auto pixbuff = static_cast<const double *>(pix.request().ptr);
        auto resbuff = static_cast<double *>(res.request().ptr);

        for(int i=0; i<size; i++)
            {
                double n_i = pixbuff[i];
                double q_i = _pocket[i];
                double from_pocket = _flush(q_i);
                double to_pocket = _fill(q_i, n_i);
                double delta = from_pocket - to_pocket;
                // std::cout << n_i << " " << q_i << " " << from_pocket << " " << to_pocket << " " << delta << std::endl;
                resbuff[i] = n_i + delta;
                _pocket[i+1] = q_i - delta;
                if (_pocket[i+1] < 0.)
                    {
                        _pocket[i+1] = 0.;
                    }
            }

        return res;
    }

    // npa  deriv(const npa & pix)
    //     {
    //         std::vector<size_t> x_shape(pix.shape(), pix.shape() + pix.ndim());
    //         py::array_t<double> res(x_shape);
    //         auto size = pix.size();
    // }


private:
    std::vector<double> _pocket;
    double _alpha, _cmax, _beta, _nmax;

};


PYBIND11_MODULE(_pocket, m) {
   py::class_<_PocketModel>(m, "_PocketModel")
	   .def(py::init<double,double,double,double>())
	   .def("apply", &_PocketModel::apply);
}

