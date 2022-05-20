// TODO: Tests, group similar functionality, make copies faster.
// TODO: Check refcounts when calling into hash and comparison functions.
// TODO: Check allocation and cleanup.


/*******************************************************************************

Our use cases differ significantly from Python's general-purpose dict type, even
when setting aside the whole immutable/grow-only and contiguous-integer-values
stuff.

What we don't care about:

  - Memory usage. Python's dicts are used literally everywhere, so a tiny
    reduction in the footprint of the average dict results in a significant gain
    for *all* Python programs. We are happy to instead trade a few extra bytes
    of RAM for a more cache-friendly hash table design. Since we don't store
    values, we are still close to the same size on average!

  - Worst-case performance. Again, Python's dicts are used for literally
    everything, so they need to be able to gracefully handle lots of hash
    collisions, whether resulting from bad hash algorithms, heterogeneous keys
    with badly-combining hash algorithms, or maliciously-formed input. We can
    safely assume that our use cases don't need to worry about these issues, and
    instead choose lookup and collision resolution strategies that utilize cache
    lines more effectively. This extends to the case of lookups for nonexistent
    keys as well; we can assume that if our users are looking for something,
    they know that it's probably there.

What we do care about:

  - Creation and update time. This is *by far* the most expensive operation you
    do on a mapping. More on this below.

  - The speed of lookups that result in hits. This is what the mapping is used
    for, so it *must* be good. More on this below.

  - Iteration order and speed. You really can't beat a Python list or tuple
    here, so we can just store the keys in one of them to avoid reinventing the
    wheel. We use a list since it allows us to grow more efficiently.

So what we need is a hash table that's easy to insert into and easy to scan.

Here's how it works. A vanilla Python dict of the form:

{a: 0, b: 1, c: 2}

...basically looks like this (assume the hashes are 15, 30, and 40):

Indices: [ 2, --, --, --, --, --,  1,  0]

Hashes:  [15, 30, 40, --, --]
Keys:    [ a,  b,  c, --, --]
Values:  [ 0,  1,  2  --, --]

It's pretty standard; keys, values, and cached hashes are stored in sequential
order, and their offsets are placed in the Indices table at position
HASH % TABLE_SIZE. Though it's not used here, collisions are resolved by jumping
around the table according to the following recurrence:

NEXT_INDEX = (5 * CURRENT_INDEX + 1 + (HASH >>= 5)) % TABLE_SIZE

This is good in the face of bad hash algorithms, but is sorta expensive. It's
also unable to utilize cache lines at all, since it's basically random (it's
literally based on random number generation)!

To contrast, the same table looks something like this for us:

Indices: [2,  --,  1,  0, --, --, --, --, --, --, --, --, --, --, --, --, --, --, --]
Hashes:  [40, --, 30, 15, --, --, --, --, --, --, --, --, --, --, --, --, --, --, --]

Keys:    [ a,  b,  c]

Right away you can see that we don't need to store the values, because they
match the indices (by design).

Notice that even though we allocated enough space in our table for 19 entries,
we still insert them into initial position HASH % 4.  This leaves the whole
15-element tail chunk of the table free for colliding keys. So, what's a good
collision-resolution strategy?

NEXT_INDEX = CURRENT_INDEX + 1

It's just a sequential scan! That means *every* collision-resolution lookup is
hot in L1 cache (and can even be predicted and speculatively executed). The
indices and hashes are actually interleaved for better cache locality as well.

We repeat this scan 15 times. We don't even have to worry about wrapping around
the edge of the table during the this part, since we've left enough free space
(equal to the number of scans) to safely run over the end. It's wasteful for a
small example like this, but for more realistic sizes it's just about perfect.

We then jump to another spot in the table using a version of the recurrence
above:

NEXT_INDEX = (5 * (CURRENT_INDEX - 15) + 1 + (HASH >>= 1)) % TABLE_SIZE

...and repeat the whole thing over again. This collision resolution strategy is
similar to what Python's sets do, so we still handle some nasty collisions and
missing keys well.

There are a couple of other tricks that we use, like globally caching integer
objects from value lookups and leaning heavily on some of the functionality that
our list of keys gives us for free, but the hardware-friendly hash table design
is what really gives us our awesome performance.

*******************************************************************************/

# include "Python.h"


/* Experimentation shows that these values work well: */

# define LOAD 0.9
# define SCAN 16

# define KEYS            'K'
# define VALUES          'V'
# define ITEMS           'I'
# define REVERSED_KEYS   'k'
# define REVERSED_VALUES 'v'
# define REVERSED_ITEMS  'i'


typedef struct {
    Py_hash_t hash;
    Py_ssize_t index;
} entry;


typedef struct {
    PyObject_VAR_HEAD
    Py_ssize_t size;
    entry *entries;
    PyObject* keys;
} AutoMapObject;


typedef struct {
    PyObject_VAR_HEAD
    char kind;
    AutoMapObject* map;
} AutoMapViewObject;


typedef struct {
    PyObject_VAR_HEAD
    char kind;
    PyObject* keys;
    Py_ssize_t index;
} AutoMapIteratorObject;


static PyTypeObject AutoMapType;
static PyTypeObject AutoMapIteratorType;
static PyTypeObject AutoMapViewType;
static PyTypeObject FrozenAutoMapType;


static PyObject* intcache = NULL;


static void
AutoMapIterator_dealloc(AutoMapIteratorObject* self)
{
    Py_DECREF(self->keys);
    Py_TYPE(self)->tp_free((PyObject*) self);
}


static AutoMapIteratorObject*
AutoMapIterator_iter(AutoMapIteratorObject* self)
{
    Py_INCREF(self);
    return self;
}


static PyObject*
AutoMapIterator_iternext(AutoMapIteratorObject* self)
{
    if (self->index == PyList_GET_SIZE(self->keys)) {
        return NULL;
    }
    PyObject *yield;
    if (self->kind == ITEMS) {
        yield = PyTuple_Pack(
            2,
            PyList_GET_ITEM(self->keys, self->index),
            PyList_GET_ITEM(intcache, self->index)
        );
    }
    else if (self->kind == KEYS) {
        yield = PyList_GET_ITEM(self->keys, self->index);
    }
    else {
        yield = PyList_GET_ITEM(intcache, self->index);
    }
    self->index++;
    Py_INCREF(yield);
    return yield;
}


static PyObject*
AutoMapIterator_methods___length_hint__(AutoMapIteratorObject* self)
{
    return PyLong_FromSsize_t(Py_MAX(0, PyList_GET_SIZE(self->keys) - self->index));
}


static PyMethodDef AutoMapIterator_methods[] = {
    {"__length_hint__", (PyCFunction) AutoMapIterator_methods___length_hint__, METH_NOARGS, NULL},
    {NULL},
};


static PyTypeObject AutoMapIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(AutoMapIteratorObject),
    .tp_dealloc = (destructor) AutoMapIterator_dealloc,
    .tp_iter = (getiterfunc) AutoMapIterator_iter,
    .tp_iternext = (iternextfunc) AutoMapIterator_iternext,
    .tp_methods = AutoMapIterator_methods,
    .tp_name = "AutoMapIterator",
};


static PyObject*
viewiter(PyObject* keys, int kind) {
    AutoMapIteratorObject* self = PyObject_New(AutoMapIteratorObject, &AutoMapIteratorType);
    if (!self) {
        return NULL;
    }
    self->kind = kind;
    self->keys = keys;
    self->index = 0;
    Py_INCREF(keys);
    return (PyObject*) self;
}


static PyObject*
AutoMapView_and(PyObject* left, PyObject* right)
{
    left = PySet_New(left);
    if (!left) {
        return NULL;
    }
    right = PySet_New(right);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject* result = PyNumber_InPlaceAnd(left, right);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyObject*
AutoMapView_or(PyObject* left, PyObject* right)
{
    left = PySet_New(left);
    if (!left) {
        return NULL;
    }
    right = PySet_New(right);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject* result = PyNumber_InPlaceOr(left, right);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyObject*
AutoMapView_subtract(PyObject* left, PyObject* right)
{
    left = PySet_New(left);
    if (!left) {
        return NULL;
    }
    right = PySet_New(right);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject* result = PyNumber_InPlaceSubtract(left, right);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyObject*
AutoMapView_xor(PyObject* left, PyObject* right)
{
    left = PySet_New(left);
    if (!left) {
        return NULL;
    }
    right = PySet_New(right);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject* result = PyNumber_InPlaceXor(left, right);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyNumberMethods AutoMapView_as_number = {
    .nb_and = (binaryfunc) AutoMapView_and,
    .nb_or = (binaryfunc) AutoMapView_or,
    .nb_subtract = (binaryfunc) AutoMapView_subtract,
    .nb_xor = (binaryfunc) AutoMapView_xor,
};


static int
AutoMapView_contains(AutoMapViewObject* self, PyObject* other)
{
    return PySequence_Contains((PyObject*) self->map, other);
}


static PySequenceMethods AutoMapView_as_sequence = {
    .sq_contains = (objobjproc) AutoMapView_contains,
};


static void
AutoMapView_dealloc(AutoMapViewObject* self)
{
    Py_DECREF(self->map);
    Py_TYPE(self)->tp_free((PyObject*) self);
}


static PyObject*
AutoMapView_iter(AutoMapViewObject* self)
{
    if (self->kind == KEYS) {
        return PyObject_GetIter(self->map->keys);
    }
    return viewiter(self->map->keys, self->kind);
}


static PyObject*
AutoMapView_methods___length_hint__(AutoMapViewObject* self)
{
    return PyLong_FromSsize_t(PyList_GET_SIZE(self->map->keys));
}


static PyObject*
AutoMapView_methods_isdisjoint(AutoMapViewObject* self, PyObject* other)
{
    PyObject* intersection = PyNumber_And((PyObject*) self, other);
    PyObject* result = PyObject_IsTrue(intersection) ? Py_True : Py_False;
    Py_DECREF(intersection);
    Py_INCREF(result);
    return result;
}


static PyMethodDef AutoMapView_methods[] = {
    {"__length_hint__", (PyCFunction) AutoMapView_methods___length_hint__, METH_NOARGS, NULL},
    {"isdisjoint", (PyCFunction) AutoMapView_methods_isdisjoint, METH_O, NULL},
    {NULL},
};


static PyObject*
AutoMapView_richcompare(AutoMapViewObject* self, PyObject* other, int op)
{
    PyObject* left = PySet_New((PyObject*) self);
    if (!left) {
        return NULL;
    }
    PyObject* right = PySet_New(other);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject* result = PyObject_RichCompare(left, right, op);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}


static PyTypeObject AutoMapViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &AutoMapView_as_number,
    .tp_as_sequence = &AutoMapView_as_sequence,
    .tp_basicsize = sizeof(AutoMapViewObject),
    .tp_dealloc = (destructor) AutoMapView_dealloc,
    .tp_iter = (getiterfunc) AutoMapView_iter,
    .tp_methods = AutoMapView_methods,
    .tp_name = "AutoMapView",
    .tp_richcompare = (richcmpfunc) AutoMapView_richcompare,
};


static PyObject*
view(AutoMapObject* map, int kind)
{
    AutoMapViewObject* self = (AutoMapViewObject*) PyObject_New(AutoMapViewObject, &AutoMapViewType);
    if (!self) {
        return NULL;
    }
    self->kind = kind;
    self->map = map;
    Py_INCREF(map);
    return (PyObject*) self;
}


static Py_ssize_t
lookup_hash(AutoMapObject* self, PyObject* key, Py_hash_t hash)
{
    PyObject *guess;
    int result;
    entry* entries = self->entries;
    Py_ssize_t mask = self->size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_hash_t h;
    Py_ssize_t i;
    Py_ssize_t stop;
    for (Py_ssize_t index = hash & mask;; index = (5 * (index - SCAN) + (mixin >>= 1) + 1) & mask) {
        for (stop = index + SCAN; index < stop; index++) {
            h = entries[index].hash;
            if (h == hash) {
                i = entries[index].index;
                guess = PyList_GET_ITEM(self->keys, i);
                if (guess == key) {
                    /* Hit. */
                    return index;
                }
                result = PyObject_RichCompareBool(guess, key, Py_EQ);
                if (result < 0) {
                    /* Error. */
                    return -1;
                }
                if (result) {
                    /* Hit. */
                    return index;
                }
            }
            else if (h == -1) {
                /* Miss. */
                return index;
            }
        }
    }
}


static Py_ssize_t
lookup(AutoMapObject* self, PyObject* key) {
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        return -1;
    }
    Py_ssize_t index = lookup_hash(self, key, hash);
    if ((index < 0) || (self->entries[index].hash == -1)) {
        return -1;
    }
    return self->entries[index].index;
}


static int
_insert(AutoMapObject* self, PyObject* key, Py_ssize_t offset, Py_hash_t hash, int append)
{
    Py_ssize_t index = lookup_hash(self, key, hash);
    if (index < 0) {
        return -1;
    }
    entry* entries = self->entries;
    if (entries[index].hash != -1) {
        PyErr_SetObject(PyExc_ValueError, key);
        return -1;
    }
    entries[index].hash = hash;
    entries[index].index = offset;
    if (append && PyList_Append(self->keys, key)) {
        entries[index].hash = -1;
        return -1;
    }
    return 0;
}


static int
insert(AutoMapObject* self, Py_ssize_t offset)
{
    PyObject* key = PyList_GET_ITEM(self->keys, offset);
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        return -1;
    }
    return _insert(self, key, offset, hash, 0);
}


static int
insert_hash(AutoMapObject* self, Py_ssize_t offset, Py_hash_t hash)
{
    return _insert(self, PyList_GET_ITEM(self->keys, offset), offset, hash, 0);
}


static int
insert_key(AutoMapObject* self, PyObject* key)
{
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        return -1;
    }
    return _insert(self, key, PyList_GET_SIZE(self->keys), hash, 1);
}


static int
fill_intcache(Py_ssize_t size)
{
    if (!intcache) {
        intcache = PyList_New(0);
        if (!intcache) {
            return -1;
        }
    }
    PyObject* item;
    for (Py_ssize_t index = PyList_GET_SIZE(intcache); index < size; index++) {
        item = PyLong_FromSsize_t(index);
        if (!item) {
            return -1;
        }
        if (PyList_Append(intcache, item)) {
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);
    }
    return 0;
}


static int
grow(AutoMapObject* self, Py_ssize_t needed)
{
    if (fill_intcache(needed)) {
        return -1;
    }
    Py_ssize_t oldsize = self->size;
    Py_ssize_t newsize = 1;
    needed /= LOAD;
    while (newsize <= needed) {
        newsize <<= 1;
    }
    if (newsize <= oldsize) {
        return 0;
    }
    entry* oldentries = self->entries;
    entry* newentries = PyMem_New(entry, newsize + SCAN - 1);
    if (!newentries) {
        return -1;
    }
    Py_ssize_t index;
    for (index = 0; index < newsize + SCAN - 1; index++) {
        newentries[index].hash = -1;
    }
    self->entries = newentries;
    self->size = newsize;
    if (oldsize) {
        for (index = 0; index < oldsize + SCAN - 1; index++) {
            if (
                (oldentries[index].hash != -1)
                && insert_hash(
                    self, oldentries[index].index, oldentries[index].hash
                )
            ) {
                PyMem_Del(self->entries);
                self->entries = oldentries;
                self->size = oldsize;
                return -1;
            }
        }
    }
    PyMem_Del(oldentries);
    return 0;
}


static AutoMapObject*
new(PyTypeObject* cls, PyObject* keys)
{
    keys = keys ? PySequence_List(keys) : PyList_New(0);
    if (!keys) {
        return NULL;
    }
    AutoMapObject* self = (AutoMapObject*) cls->tp_alloc(cls, 0);
    if (!self) {
        Py_DECREF(keys);
        return NULL;
    }
    self->keys = keys;
    Py_ssize_t size = PyList_GET_SIZE(keys);
    if (grow(self, size)) {
        Py_DECREF(self);
        return NULL;
    }
    for (Py_ssize_t index = 0; index < size; index++) {
        if (insert(self, index)) {
            Py_DECREF(self);
            return NULL;
        }
    }
    return self;
}


static int
extend(AutoMapObject* self, PyObject* keys)
{
    keys = PySequence_Fast(keys, "expected an iterable of keys.");
    if (!keys) {
        return -1;
    }
    Py_ssize_t extendsize = PySequence_Fast_GET_SIZE(keys);
    if (grow(self, PyList_GET_SIZE(self->keys) + extendsize)) {
        Py_DECREF(keys);
        return -1;
    }
    for (Py_ssize_t index = 0; index < extendsize; index++) {
        if (insert_key(self, PySequence_Fast_GET_ITEM(keys, index))) {
            Py_DECREF(keys);
            return -1;
        }
    }
    Py_DECREF(keys);
    return 0;
}


static int
append(AutoMapObject* self, PyObject* key)
{
    if (grow(self, PyList_GET_SIZE(self->keys) + 1)) {
        return -1;
    }
    if (insert_key(self, key)) {
        return -1;
    }
    return 0;
}


static Py_ssize_t
AutoMap_length(AutoMapObject* self)
{
    return PyList_GET_SIZE(self->keys);
}


static PyObject*
get(AutoMapObject* self, PyObject* key, PyObject* missing) {
    Py_ssize_t result = lookup(self, key);
    if (result < 0) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        if (missing) {
            Py_INCREF(missing);
            return missing;
        }
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }
    PyObject *index = PyList_GET_ITEM(intcache, result);
    Py_INCREF(index);
    return index;
}


static PyObject*
AutoMap_subscript(AutoMapObject* self, PyObject* key)
{
    return get(self, key, NULL);
}


static PyMappingMethods AutoMap_as_mapping = {
    .mp_length = (lenfunc) AutoMap_length,
    .mp_subscript = (binaryfunc) AutoMap_subscript,
};


static PyObject*
AutoMap_or(PyObject* left, PyObject* right)
{
    if (
        !PyObject_TypeCheck(left, &FrozenAutoMapType)
        || !PyObject_TypeCheck(right, &FrozenAutoMapType)
    ) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    AutoMapObject *updated = new(Py_TYPE(left), left);
    if (!updated) {
        return NULL;
    }
    if (extend(updated, right)) {
        Py_DECREF(updated);
        return NULL;
    }
    return (PyObject*) updated;
}


static PyNumberMethods FrozenAutoMap_as_number = {
    .nb_or = (binaryfunc) AutoMap_or,
};


static int
AutoMap_contains(AutoMapObject* self, PyObject* key)
{
    Py_ssize_t result = lookup(self, key);
    if (result < 0) {
        if (PyErr_Occurred()) {
            return -1;
        }
        return 0;
    }
    return 1;
}


static PySequenceMethods AutoMap_as_sequence = {
    .sq_contains = (objobjproc) AutoMap_contains,
};


static void
AutoMap_dealloc(AutoMapObject* self)
{
    PyMem_Del(self->entries);
    Py_XDECREF(self->keys);
    Py_TYPE(self)->tp_free((PyObject*) self);
}


static Py_hash_t
FrozenAutoMap_hash(AutoMapObject* self)
{
    PyObject *tuple = PyList_AsTuple(self->keys);
    if (!tuple) {
        return -1;
    }
    Py_hash_t hash = PyObject_Hash(tuple);
    Py_DECREF(tuple);
    return hash;
}


static PyObject*
AutoMap_iter(AutoMapObject* self)
{
    // return viewiter(self->keys, KEYS);
    return PyObject_GetIter(self->keys);
}


static PyObject*
AutoMap_methods___getnewargs__(AutoMapObject* self)
{
    PyObject *keys = PyList_AsTuple(self->keys);
    if (!keys) {
        return NULL;
    }
    PyObject *pickled = PyTuple_Pack(1, keys);
    Py_DECREF(keys);
    return pickled;
}


static PyObject*
AutoMap_methods___reversed__(AutoMapObject* self)
{
    return PyObject_CallMethod(self->keys, "__reversed__", NULL);
}


static PyObject*
AutoMap_methods___sizeof__(AutoMapObject* self)
{
    PyObject *listsizeof = PyObject_CallMethod(self->keys, "__sizeof__", NULL);
    if (!listsizeof) {
        return NULL;
    }
    Py_ssize_t listbytes = PyLong_AsSsize_t(listsizeof);
    Py_DECREF(listsizeof);
    if (listbytes == -1 && PyErr_Occurred()) {
        return NULL;
    }
    return PyLong_FromSsize_t(
        Py_TYPE(self)->tp_basicsize
        + listbytes
        + (self->size + SCAN - 1) * sizeof(entry)
    );
}


static PyObject*
AutoMap_methods_get(AutoMapObject* self, PyObject* args)
{
    PyObject *key, *missing = Py_None;
    if (!PyArg_UnpackTuple(args, Py_TYPE(self)->tp_name, 1, 2, &key, &missing)) {
        return NULL;
    }
    return get(self, key, missing);
}


static PyObject*
AutoMap_methods_items(AutoMapObject* self)
{
    return view(self, ITEMS);
}


static PyObject*
AutoMap_methods_keys(AutoMapObject* self)
{
    return view(self, KEYS);
}


static PyObject*
AutoMap_methods_values(AutoMapObject* self)
{
    return view(self, VALUES);
}


static PyMethodDef FrozenAutoMap_methods[] = {
    {"__getnewargs__", (PyCFunction) AutoMap_methods___getnewargs__, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) AutoMap_methods___reversed__, METH_NOARGS, NULL},
    {"__sizeof__", (PyCFunction) AutoMap_methods___sizeof__, METH_NOARGS, NULL},
    {"get", (PyCFunction) AutoMap_methods_get, METH_VARARGS, NULL},
    {"items", (PyCFunction) AutoMap_methods_items, METH_NOARGS, NULL},
    {"keys", (PyCFunction) AutoMap_methods_keys, METH_NOARGS, NULL},
    {"values", (PyCFunction) AutoMap_methods_values, METH_NOARGS, NULL},
    {NULL},
};


static PyObject*
AutoMap_new(PyTypeObject* cls, PyObject* args, PyObject* kwargs)
{
    const char *name = cls->tp_name;
    if (kwargs) {
        PyErr_Format(PyExc_TypeError, "%s takes no keyword arguments", name);
        return NULL;
    }
    PyObject *keys = NULL;
    if (!PyArg_UnpackTuple(args, name, 0, 1, &keys)) {
        return NULL;
    }
    return (PyObject*) new(cls, keys);
}


static PyObject*
AutoMap_repr(AutoMapObject* self)
{
    const char *name = Py_TYPE(self)->tp_name;
    if (PyList_GET_SIZE(self->keys)) {
        return PyUnicode_FromFormat("%s(%R)", name, self->keys);
    }
    return PyUnicode_FromFormat("%s()", name);
}


static PyObject*
AutoMap_richcompare(AutoMapObject* self, PyObject* other, int op)
{
    if ((op != Py_EQ) && (op != Py_NE)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if ((PyObject*) self == other) {
        return PyBool_FromLong(op == Py_EQ);
    }
    int subclass = PyObject_TypeCheck(other, &FrozenAutoMapType);
    if (subclass < 0) {
        return NULL;
    }
    if (!subclass) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return PyObject_RichCompare(self->keys, ((AutoMapObject*) other)->keys, op);
}


static PyTypeObject FrozenAutoMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_mapping = &AutoMap_as_mapping,
    .tp_as_number = &FrozenAutoMap_as_number,
    .tp_as_sequence = &AutoMap_as_sequence,
    .tp_basicsize = sizeof(AutoMapObject),
    .tp_dealloc = (destructor) AutoMap_dealloc,
    .tp_doc = "An immutable autoincremented integer-valued mapping.",
    .tp_hash = (hashfunc) FrozenAutoMap_hash,
    .tp_iter = (getiterfunc) AutoMap_iter,
    .tp_methods = FrozenAutoMap_methods,
    .tp_name = "automap.FrozenAutoMap",
    .tp_new = AutoMap_new,
    .tp_repr = (reprfunc) AutoMap_repr,
    .tp_richcompare = (richcmpfunc) AutoMap_richcompare,
};


static PyObject*
AutoMap_inplace_or(AutoMapObject* self, PyObject* other)
{
    if (extend(self, other)) {
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject*) self;
}


static PyNumberMethods AutoMap_as_number = {
    .nb_inplace_or = (binaryfunc) AutoMap_inplace_or,
};


static PyObject*
AutoMap_methods_add(AutoMapObject* self, PyObject* other)
{
    if (append(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject*
AutoMap_methods_update(AutoMapObject* self, PyObject* other)
{
    if (extend(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyMethodDef AutoMap_methods[] = {
    {"add", (PyCFunction) AutoMap_methods_add, METH_O, NULL},
    {"update", (PyCFunction) AutoMap_methods_update, METH_O, NULL},
    {NULL},
};


static PyTypeObject AutoMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &AutoMap_as_number,
    .tp_base = &FrozenAutoMapType,
    .tp_doc = "A grow-only autoincremented integer-valued mapping.",
    .tp_methods = AutoMap_methods,
    .tp_name = "automap.AutoMap",
    .tp_richcompare = (richcmpfunc) AutoMap_richcompare,
};


/* automap **********************************************************/


void
automap_free(PyObject* self)
{
    Py_CLEAR(intcache);
};


static struct PyModuleDef automap_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_doc = "High-performance autoincremented integer-valued mappings.",
    .m_free = (freefunc) automap_free,
    .m_name = "automap",
    .m_size = sizeof(PyObject*),
};


PyObject*
PyInit_automap(void)
{
    PyObject* automap = PyModule_Create(&automap_module);
    if (
        !automap
        || PyType_Ready(&AutoMapType)
        || PyType_Ready(&AutoMapIteratorType)
        || PyType_Ready(&AutoMapViewType)
        || PyType_Ready(&FrozenAutoMapType)
        || PyModule_AddObject(automap, "AutoMap", (PyObject*) &AutoMapType)
        || PyModule_AddObject(automap, "FrozenAutoMap", (PyObject*) &FrozenAutoMapType)
    ) {
        return NULL;
    }
    return automap;
}
