
def check_array(name, arr, shapes=None, dtype=None):
    # check dtype
    if dtype is not None:
        assert arr.dtype == dtype, f"'{name}' has invalid dtype '{arr.dtype}' should be '{dtype}'"

    # check shapes
    if shapes is not None:
        match_shape = False
        str_shape = ""
        for shape in shapes:
            assert isinstance(shape, (list, tuple))
            # generate string of correct shape
            if len(str_shape) > 0:
                str_shape += " or "
            str_shape += "["
            for d, y in enumerate(shape):
                if d > 0:
                    str_shape += ","
                if y is None:
                    str_shape += ">0"
                elif isinstance(y, int):
                    str_shape += str(y)
                elif isinstance(y, (list, tuple)):
                    str_shape += "|".join(str(v) for v in y)
                else:
                    raise NotImplementedError()
            str_shape += "]"
            # check dims
            if len(arr.shape) != len(shape):
                continue
            # check match
            good = True
            for x, y in zip(arr.shape, shape):
                if y is None or y == -1:
                    continue
                elif isinstance(y, int):
                    good &= x == y
                elif isinstance(y, (list, tuple)):
                    good &= x in y
                else:
                    raise NotImplementedError()
            if good:
                match_shape = True
                break
        assert match_shape, f"'{name}' has invalid shape '{list(arr.shape)}', should be '{str_shape}'"
