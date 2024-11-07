from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, parameters: dict = None, default_params: dict = None):
        """
        Initialize the model with default and user-provided parameters.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of user-provided parameters.
        default_params : dict, optional
            Dictionary of default parameters in the format:
            {'variable_name': [default_value, type, validator(optional)]}
        """

        # Keep sets for what problems each variables is for
        self.__regression_attributes = set()
        self.__classification_attributes = set()

        # Set default parameters
        self._set_default_parameters(default_params)

        # Take in optional user input for parameters
        if parameters is not None:
            self._update_parameters(parameters)

    def _set_default_parameters(self, default_params: dict):
        """
        Set default parameters based on the default_params dictionary.

        Parameters
        ----------
        default_params : dict
            Requires default parameters to be in the following format:
            {'variable_name': [default_value,
                type(Regression/Classification/Both), validator(optional)]}
        """
        for param, (default_value, prob_type, validator) in default_params.items():
            # Create the variable
            setattr(self, f"_{param}", default_value)

            # Define a wrapper function to generate getter and setter
            def create_property_functions(param_name, validator):
                def getter(self):
                    return getattr(self, f"_{param_name}")

                def setter(self, value):
                    if validator:
                        assert validator(value)
                    setattr(self, f"_{param_name}", value)

                return getter, setter

            # Get getter and setter functions from the wrapper function
            getter, setter = create_property_functions(param, validator)

            # Dynamically create attributes with getters/setters for each parameter
            setattr(self.__class__, param, property(fget=getter, fset=setter))

            # Set the variable type for problems
            if prob_type == "BOTH":
                self.__regression_attributes.add(param)
                self.__classification_attributes.add(param)
            elif prob_type == "REGRESSION":
                self.__regression_attributes.add(param)
            elif prob_type == "CLASSIFICATION":
                self.__classification_attributes.add(param)
            elif prob_type == "NONE":
                continue
            else:
                raise ValueError(
                    f"Please provide a valid problem type for {param}: "
                    + "BOTH/REGRESSION/CLASSIFICATION/NONE"
                )

    def _update_parameters(self, parameters: dict):
        """
        Update model parameters based on user-provided dictionary.

        Parameters
        ----------
        parameters : dict
            Dictionary of user-provided parameters.
        """
        for key, value in parameters.items():
            protected_key = f"_{key}"
            if hasattr(self, protected_key):
                setattr(self, protected_key, value)
            else:
                raise NameError(
                    f"{protected_key} "
                    + f"is not a valid variable for {self.__class__.__name__}"
                )

    def _get_regression_model(self, model):
        """
        Returns the Regression model with the given parameters.

        Parameters
        ----------
        model : class
            The model class to instantiate with regression parameters.

        Returns
        -------
        object
            Instantiated regression model.
        """
        model_params = {
            attr: getattr(self, attr) for attr in self.__regression_attributes
        }
        return model(**model_params)

    def _get_classification_model(self, model):
        """
        Returns the Classification model with the given parameters.

        Parameters
        ----------
        model : class
            The model class to instantiate with classification parameters.

        Returns
        -------
        object
            Instantiated classification model.
        """
        model_params = {
            attr: getattr(self, attr) for attr in self.__classification_attributes
        }
        return model(**model_params)

    @abstractmethod
    def regressor(self):
        """
        Abstract method to define and return a model based on the parameters set in the
        class and problem type.

        This method should be implemented by subclasses to specify the model
        architecture and return an instantiated model object that is ready for training
        or evaluation based on problem type.
        """
        pass
