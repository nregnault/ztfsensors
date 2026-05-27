"""
Model factory for equilibrium models.

This module provides a factory pattern for registering and instantiating
equilibrium model classes. The factory enables seamless serialization and
deserialization of models by maintaining a registry of model types keyed
by their name and version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseEquilibriumModel


class EquilibriumModelFactory:
    """
    Factory for creating and registering equilibrium model classes.

    This factory maintains a global registry of model classes, allowing models
    to be instantiated from serialized headers without hardcoding the model type.
    Models are registered using the @EquilibriumModelFactory.register decorator.

    The registry is keyed by (model_name, model_version) tuples to ensure that
    different versions of the same model can coexist.

    Attributes
    ----------
    _registry : dict[tuple[str, str], type[BaseEquilibriumModel]]
        Global registry mapping (model_name, model_version) to model classes.

    Examples
    --------
    Register a new model class::

        @EquilibriumModelFactory.register
        class MyModel(BaseEquilibriumModel):
            MODEL_NAME = "my_model"
            MODEL_VERSION = "v1"
            ...

    Instantiate a model from a header::

        header = {"model_name": "my_model", "model_version": "v1", ...}
        model = EquilibriumModelFactory.from_header(header)
    """

    _registry: dict[tuple[str, str], type["BaseEquilibriumModel"]] = {}

    @classmethod
    def register(
        cls, model_cls: type["BaseEquilibriumModel"]
    ) -> type["BaseEquilibriumModel"]:
        """
        Register a model class in the factory's global registry.

        This method is designed to be used as a class decorator. It registers
        the model class so it can be instantiated later from serialized headers.

        Parameters
        ----------
        model_cls : type[BaseEquilibriumModel]
            The model class to register. Must have MODEL_NAME and MODEL_VERSION
            class attributes defined.

        Returns
        -------
        type[BaseEquilibriumModel]
            The same model class (allows use as a decorator).

        Raises
        ------
        ValueError
            If a model with the same (MODEL_NAME, MODEL_VERSION) is already registered.

        Examples
        --------
        ::

            @EquilibriumModelFactory.register
            class MyModel(BaseEquilibriumModel):
                MODEL_NAME = "my_model"
                MODEL_VERSION = "v1"
                ...
        """
        key = (model_cls.MODEL_NAME, model_cls.MODEL_VERSION)
        if key in cls._registry:
            raise ValueError(f"Model already registered: {key}")
        cls._registry[key] = model_cls
        return model_cls

    @classmethod
    def from_header(cls, header: dict[str, Any]) -> "BaseEquilibriumModel":
        """
        Instantiate a model from a serialized header dictionary.

        Looks up the appropriate model class in the registry based on the
        model_name and model_version fields in the header, then delegates
        to that class's from_header() method for instantiation.

        Parameters
        ----------
        header : dict[str, Any]
            Dictionary containing at minimum 'model_name' and 'model_version'
            keys, plus any additional fields required by the specific model.

        Returns
        -------
        BaseEquilibriumModel
            Instance of the appropriate model class with parameters from the header.

        Raises
        ------
        ValueError
            If the (model_name, model_version) combination is not found in the
            registry, or if the model class's from_header() method raises an error.

        Examples
        --------
        ::

            header = {
                "model_name": "spline_temp_eq",
                "model_version": "v1",
                "basis_grid": [50.0, 100.0, 500.0],
                "temp_deg": 5,
                ...
            }
            model = EquilibriumModelFactory.from_header(header)
        """
        model_name = header.get("model_name")
        model_version = header.get("model_version")
        if model_name is None or model_version is None:
            raise ValueError(
                f"Header must contain 'model_name' and 'model_version', "
                f"got model_name={model_name}, model_version={model_version}"
            )
        key = (model_name, model_version)
        try:
            model_cls = cls._registry[key]
        except KeyError as exc:
            raise ValueError(f"Unknown model: {key}") from exc
        return model_cls.from_header(header)
