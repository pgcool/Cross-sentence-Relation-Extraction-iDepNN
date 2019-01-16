
import theano
from theano import tensor
from collections import OrderedDict

def pack(arg):
    """Pack variables into a list.
    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. Lists will be
        returned as is, and tuples will be cast to lists. Any other
        variable will be returned in a singleton list.
    Returns
    -------
    list
        List containing the arguments
    """
    if isinstance(arg, (list, tuple)):
        return list(arg)
    else:
        return [arg]


def shared_floatx(value, name=None, borrow=False, dtype=None):
    """Transform a value into a shared variable of type floatX.
    Parameters
    ----------
    value : :class:`~numpy.ndarray`
        The value to associate with the Theano shared.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    borrow : :obj:`bool`, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : :obj:`str`, optional
        The `dtype` of the shared variable. Default value is
        :attr:`config.floatX`.
    Returns
    -------
    :class:`tensor.TensorSharedVariable`
        A Theano shared variable with the requested value and `dtype`.
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def named_copy(variable, new_name):
    """Clones a variable and set a new name to the clone."""
    result = variable.copy()
    result.name = new_name
    return result


def dict_subset(dict_, keys, pop=False, must_have=True):
    """Return a subset of a dictionary corresponding to a set of keys.
    Parameters
    ----------
    dict_ : dict
        The dictionary.
    keys : iterable
        The keys of interest.
    pop : bool
        If ``True``, the pairs corresponding to the keys of interest are
        popped from the dictionary.
    must_have : bool
        If ``True``, a ValueError will be raised when trying to retrieve a
        key not present in the dictionary.
    Returns
    -------
    result : ``OrderedDict``
        An ordered dictionary of retrieved pairs. The order is the same as
        in the ``keys`` argument.
    """
    not_found = object()

    def extract(k):
        if pop:
            if must_have:
                return dict_.pop(k)
            return dict_.pop(k, not_found)
        if must_have:
            return dict_[k]
        return dict_.get(k, not_found)

    result = [(key, extract(key)) for key in keys]
    return OrderedDict([(k, v) for k, v in result if v is not not_found])