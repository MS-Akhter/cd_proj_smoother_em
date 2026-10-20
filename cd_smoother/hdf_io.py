"""HDF5 input/output utilities for the simulation results (Sec. V experiments)."""
import pathlib
from datetime import timedelta
from typing import Any, Optional, Union

import h5py
import jax.numpy as jnp
import numpy as np
from jaxtyping import Shaped


def save_to_hdf(
    file_name: Union[str, pathlib.Path],
    variable_dict: dict[str, Any],
    classes: list[type] = [],
    excluded_vars: Optional[list[str]] = None,
) -> None:
    """Save the variables of a dictionary to an HDF5 file.

    Parameters
    ----------
    file_name : str or pathlib.Path
        File name of the HDF5 file (overwritten if it exists).
    variable_dict : dict[str, Any]
        Variables to save; scalars, strings, timedeltas and numeric
        ``Shaped[np.ndarray, "..."]`` / ``Shaped[jnp.ndarray, "..."]`` arrays
        are stored as datasets.
    classes : list[type], optional
        Object types that are saved recursively as HDF5 groups.
    excluded_vars : list[str], optional
        Variable names to skip.

    Returns
    -------
    None
    """

    try:
        with h5py.File(file_name, 'w') as f:
            __save_object(f, variable_dict, classes, excluded_vars=excluded_vars)
    except AttributeError as e:
        print("Something wrong here ", e.__class__, " occured.")
        print("Full detail ", e.__cause__)


def __save_object(
    f: Union[h5py.File, h5py.Group],
    obj: Any,
    classes: list[type] = [],
    end_here: bool = False,
    excluded_vars: Optional[list[str]] = None,
) -> None:
    """Write the members of `obj` into an open HDF5 file or group.

    Parameters
    ----------
    f : h5py.File or h5py.Group
        Open, writable HDF5 container.
    obj : dict, list, tuple or object
        Container whose items (or ``__dict__`` attributes) are saved.
    classes : list[type], optional
        Object types that are saved recursively as HDF5 groups.
    end_here : bool, optional
        If True, do not recurse into members of type `classes`.
    excluded_vars : list[str], optional
        Member names to skip.

    Returns
    -------
    None
    """
    if excluded_vars is None:
        excluded_vars = []
    basic_data_types = [int, float, str, bool]
    if isinstance(obj, dict):
        dict_ = obj
    elif isinstance(obj, list):
        name_ = [str(i_) for i_ in range(len(obj))]
        dict_ = dict(zip(name_,
                         obj))
    elif isinstance(obj, tuple):
        name_ = [str(i_) for i_ in range(len(obj))]
        dict_ = dict(zip(name_,
                         obj))
    else:
        dict_ = obj.__dict__

    for key, value in dict_.items():
        if key in excluded_vars:
            continue
        if __is_instance(value, basic_data_types):
            f.create_dataset(key, data=value)
            continue
        if __is_instance(value, [timedelta, ]):
            f.create_dataset(key, data=value.total_seconds())
            continue

        elif __is_instance(value, [jnp.ndarray, np.ndarray]):
            if value.dtype in [jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64, jnp.complex128,
                               np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]:
                if value.ndim > 0:
                    f.create_dataset(key, data=value, compression='gzip')
                else:
                    f.create_dataset(key, data=value)
                continue
        else:
            if not end_here:
                if __is_instance(value, classes):
                    grp = f.create_group(key)
                    __save_object(grp, value, classes, excluded_vars=excluded_vars)


def __is_instance(value: Any, classes: list[type]) -> bool:
    """Return True if `value` is an instance of any of `classes`.

    Parameters
    ----------
    value : Any
        Object to test.
    classes : list[type]
        Candidate types.

    Returns
    -------
    bool
        Whether ``isinstance(value, c)`` holds for some ``c`` in `classes`.
    """
    it_is = False
    for a_type in classes:
        it_is = isinstance(value, a_type)
        if it_is:
            break
    return it_is


def __is_list_of_strings_contains_a_string(a_string: str, list_of_strings: list[str]) -> bool:
    """Return True if any element of `list_of_strings` is a substring of `a_string`.

    Parameters
    ----------
    a_string : str
        String to search in.
    list_of_strings : list[str]
        Candidate substrings.

    Returns
    -------
    bool
        Whether some element of `list_of_strings` occurs in `a_string`.
    """
    it_is = False
    for another_string in list_of_strings:
        it_is = (another_string in a_string)
        if it_is:
            break
    return it_is


def get_variables(
    file: h5py.File, data_set_names: list[str], array_names: list[str]
) -> dict[str, Optional[Shaped[np.ndarray, "..."]]]:
    """Read datasets from an open HDF5 file.

    Parameters
    ----------
    file : h5py.File
        Open, readable HDF5 file.
    data_set_names : list[str]
        Dataset paths inside `file`.
    array_names : list[str]
        Keys of the returned dictionary, aligned with `data_set_names`.

    Returns
    -------
    dict[str, Shaped[np.ndarray, "..."] or None]
        Dataset contents; None (with a warning) for missing datasets.
    """
    variables = {}
    for i in range(len(data_set_names)):
        try:
            variables[array_names[i]] = file[data_set_names[i]][()]
        except KeyError:
            print("Warning, dataset with name = {} does not exists in this file!".format(data_set_names[i]))
            variables[array_names[i]] = None
    return variables


def load_hdf_file(
    file_name: str, relative_path: str, data_set_names: list[str], array_names: list[str]
) -> dict[str, Optional[Shaped[np.ndarray, "..."]]]:
    """Load selected datasets from an HDF5 file.

    Parameters
    ----------
    file_name : str
        File name inside `relative_path`.
    relative_path : str
        Directory containing the file.
    data_set_names : list[str]
        Dataset paths inside the file.
    array_names : list[str]
        Keys of the returned dictionary, aligned with `data_set_names`.

    Returns
    -------
    dict[str, Shaped[np.ndarray, "..."] or None]
        Dataset contents; None for missing datasets.

    Raises
    ------
    FileNotFoundError
        If `relative_path` does not exist.
    """
    simulation_results_path = pathlib.Path(relative_path)
    if not simulation_results_path.exists():
        raise FileNotFoundError

    file = simulation_results_path / file_name
    with h5py.File(file, mode='r') as f:
        array_dict = get_variables(f, data_set_names, array_names)
    return array_dict


def override_a_dataset(file: pathlib.Path, data_set_name: str, new_value: Any) -> None:
    """Overwrite the contents of an existing dataset in place.

    Parameters
    ----------
    file : pathlib.Path
        HDF5 file, opened in read/write mode.
    data_set_name : str
        Path of the existing dataset.
    new_value : Any
        New value; must be broadcastable to the dataset's shape and dtype.

    Returns
    -------
    None
    """
    with h5py.File(file, 'r+') as f:
        current_value = f[data_set_name]
        current_value[...] = new_value
