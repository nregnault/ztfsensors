"""
Tests for ZTF Sensors analysis package.

This package contains centralized unit tests for all ztfsensors modules.

Pocket effect equilibrium models:
- test_base.py: Abstract base classes (JaxEqFunc, BaseEquilibriumModel)
- test_factory.py: Model factory and registry
- test_poly_temp_eq_model.py: Polynomial temperature equilibrium model
- test_spline_temp_eq_model.py: Spline temperature equilibrium model
- test_db.py: Database for equilibrium function parameters
- test_fit.py: Fitting utilities and diagnostics

Other modules:
- (Future tests for other ztfsensors components)

Common fixtures and utilities are defined in conftest.py.
"""
